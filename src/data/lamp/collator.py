import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy

from ...arguments import CompressionArguments
from .retriever import BaseRetriever
from .utils import generate_instruction, split_input, ppep_list, classification_candidates
from ..collator import prepare_comp_attn_mask_llama

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BaseCollatorForLaMP:
    tokenizer: PreTrainedTokenizerBase
    retriever: BaseRetriever
    comp_args: CompressionArguments
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    comp_token: int = 32100
    sum_token: Optional[int] = None
    pad_token: int = 0
    k: int = 16
    model_type: str = "llama"

    def get_candidate_batch(self, lamp_index):
        candidates = classification_candidates[lamp_index]
        inputs = "Output:" if self.model_type == "llama" else ""

        inputs = self.get_tokenized_inputs(inputs)
        candidates = self.get_tokenized_completion("", candidates=candidates)
        model_inputs = defaultdict(list)

        for c in candidates:
            model_inputs["input_ids"].append(inputs + c)
            model_inputs["labels"].append([self.label_pad_token_id] * len(inputs) + c)

        model_inputs["input_ids"] = torch.fliplr(
            pad_sequence([torch.tensor(v[::-1]) for v in model_inputs["input_ids"]],
                         padding_value=self.tokenizer.pad_token_id,
                         batch_first=True))
        model_inputs["labels"] = torch.fliplr(
            pad_sequence([torch.tensor(v[::-1]) for v in model_inputs["labels"]],
                         padding_value=self.label_pad_token_id,
                         batch_first=True))

        return model_inputs

    def __post_init__(self):
        self.max_length = 1024
        self.max_length_with_comp_token = self.max_length + self.comp_args.num_comp_tokens * self.k

        if self.comp_args.attn_type in ["merge"]:
            self.max_length_with_comp_token += self.comp_args.num_comp_tokens

        if self.sum_token is None and hasattr(self.tokenizer, "sum_token_id"):
            self.sum_token = self.tokenizer.sum_token_id

    def length_without_comp_tokens(self, input_ids, comp_ids):
        if comp_ids is None:
            return len(input_ids)
        return len([i for i in input_ids if i not in comp_ids])

    def generate_inputs(self, profile, lamp_index, input_, instruction, maybe_comp_str):
        profile_processed = ppep_list(profile, lamp_index)

        comp_profile = self.comp_args.comp_type == "fixed"
        recur_profile = self.comp_args.comp_type == "online"

        if self.comp_args.attn_type in ["merge"]:
            num_sum_toks = len(self.tokenizer.sum_token_id)
            sum_tok_str = "".join([f"<COMP{i}>" for i in range(num_sum_toks, 2 * num_sum_toks)])
        elif self.comp_args.attn_type in ["merge_recur"]:
            num_sum_toks = len(self.tokenizer.sum_token_id)
            maybe_comp_str += "".join([f"<COMP{i}>" for i in range(num_sum_toks, 2 * num_sum_toks)])
            sum_tok_str = ""
        else:
            num_sum_toks = 0
            sum_tok_str = ""

        instruction = generate_instruction(instruction, profile_processed, lamp_index,
                                            maybe_comp_str=maybe_comp_str,
                                            sum_tok_str=sum_tok_str,
                                            gist_profile=comp_profile,
                                            recur_profile=recur_profile)

        if input_:
            source = f"Instruction: {instruction}\nInput: {input_}"
        else:
            source = f"Instruction: {instruction}\n{maybe_comp_str}"

        if self.model_type == "llama":
            source += "\nOutput:"


        return source

    def get_tokenized_completion(self, output, candidates=None):

        if candidates is None:
            tokenized_completion = self.tokenizer(
                output, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        else:
            tokenized_completion = self.tokenizer(candidates, add_special_tokens=False)["input_ids"]
            tokenized_completion = [t + [self.tokenizer.eos_token_id] for t in tokenized_completion]

        return tokenized_completion

    def get_max_length_for_input(self, output, candidates=None):
        if candidates is None:
            max_length_for_input = self.max_length - len(
                self.get_tokenized_completion(output, candidates))
        else:
            tokenized_candidates = self.get_tokenized_completion(output, candidates)
            completion_len_arr = np.array([len(i) for i in tokenized_candidates])
            longest_completion = tokenized_candidates[np.argmax(completion_len_arr)]
            max_length_for_input = self.max_length - len(longest_completion)

        return max_length_for_input

    def get_tokenized_inputs(self, inputs):
        tokenized_inputs = self.tokenizer(inputs, max_length=2048)["input_ids"]
        return tokenized_inputs

    def process_batch_instance(self, instance, candidates=None):
        eval = not self.model.training
        if not self.comp_args.add_comp_token:
            # Add comp tokens later, during tokenization.
            maybe_comp_str = ""
        else:
            # attn_type: comp, merge, concat, no
            # maybe_comp_str = "<comp>" * self.num_comp_tokens
            if self.comp_args.num_comp_tokens == 1 and self.comp_args.attn_type not in [
                    "merge", "merge_recur"
            ]:
                maybe_comp_str = "<COMP>"
            else:
                maybe_comp_str = "".join(
                    [f"<COMP{i}>" for i in range(self.comp_args.num_comp_tokens)])

        comp_ids = self.tokenizer.comp_token_id

        if type(comp_ids) == int:
            comp_ids = [comp_ids]

        input_ = instance["input"]
        output = instance["output"]
        profile = instance["profile"]
        lamp_index = instance["lamp_index"]

        max_length_for_input = self.get_max_length_for_input(output, candidates)

        # dict(key: list<val>) -> list(dict(key: val))
        n_profile = len(list(profile.values())[0])
        profile = [{k: v[i] for k, v in profile.items()} for i in range(n_profile)]
        instruction, input_ = split_input(input_, lamp_index)

        profile = self.retriever(input_, profile, eval=eval)

        inputs = self.generate_inputs(profile, lamp_index, input_, instruction, maybe_comp_str)

        comp_ids = self.tokenizer.comp_token_id
        if type(comp_ids) == int:
            comp_ids = [comp_ids]

        tokenized_inputs = self.get_tokenized_inputs(inputs)
        if self.length_without_comp_tokens(tokenized_inputs, comp_ids) > max_length_for_input:
            while (self.length_without_comp_tokens(tokenized_inputs, comp_ids) >
                   max_length_for_input):
                profile = profile[:-1]  # Pop out last profile entry

                inputs = self.generate_inputs(profile, lamp_index, input_, instruction,
                                              maybe_comp_str)
                tokenized_inputs = self.get_tokenized_inputs(inputs)

                if len(profile) == 0:
                    tokenized_inputs = tokenized_inputs[:max_length_for_input]
                    break

        tokenized_completions = self.get_tokenized_completion(output, candidates)
        if candidates is None:
            results = {
                "tokenized_inputs": tokenized_inputs,
                "tokenized_completions": tokenized_completions,
                "labels": [self.label_pad_token_id] * len(tokenized_inputs) + tokenized_completions
            }
        else:

            labels = []
            for tokenized_completion in tokenized_completions:
                label = [self.label_pad_token_id] * len(tokenized_inputs) + tokenized_completion
                labels.append(label)

            results = {
                "tokenized_inputs": tokenized_inputs,
                "tokenized_completions": tokenized_completions,
                "labels": labels,
                "answers": candidates.index(output),
            }

        return results

    def __call__(self, batch, return_tensors=None):
        raise NotImplementedError()

    def collate_for_classification(self, batch, return_tensors=None):
        raise NotImplementedError()


@dataclass
class LLaMACollatorForLaMP(BaseCollatorForLaMP):
    """Data collator for decoder-only models. Does left padding."""
    def __post_init__(self):
        super().__post_init__()
        self.model_type = "llama"

    def pad_inputs(self, model_inputs):
        # Left-pad inputs, convert to tensor.
        for key, value in model_inputs.items():
            if key == "labels":
                pad_token_id = self.label_pad_token_id
            else:
                pad_token_id = self.tokenizer.pad_token_id
            # To left-pad inputs, reverse, then right-pad, then reverse.
            # Pad attention too (with pad_token_id)
            value_tensors = [torch.tensor(v[::-1]) for v in value]
            model_inputs[key] = torch.fliplr(
                pad_sequence(
                    value_tensors,
                    batch_first=True,
                    padding_value=pad_token_id,
                ))

    def __call__(self, batch, return_tensors=None):

        model_inputs = defaultdict(list)
        for instance in batch:

            results = self.process_batch_instance(instance)
            tokenized_prompt = results["tokenized_inputs"]
            tokenized_completion = results["tokenized_completions"]
            tokenized_source = tokenized_prompt + tokenized_completion
            labels = results["labels"]

            model_inputs["input_ids"].append(tokenized_source)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1 for _ in tokenized_source])

            model_inputs["prompt_input_ids"].append(tokenized_prompt)
            model_inputs["prompt_attention_mask"].append([1 for _ in tokenized_prompt])

            model_inputs["completion_input_ids"].append(tokenized_completion)
            model_inputs["completion_attention_mask"].append([1 for _ in tokenized_completion])

        self.pad_inputs(model_inputs)
        model_inputs = prepare_comp_attn_mask_llama(model_inputs, self.comp_args, self.comp_token,
                                                    self.sum_token, self.pad_token)

        return model_inputs

    def collate_for_classification(self, batch, return_tensors=None):
        model_inputs = defaultdict(list)
        answers = []
        lamp_index = batch[0]["lamp_index"]
        candidates = classification_candidates[lamp_index]
        for instance in batch:

            results = self.process_batch_instance(instance, candidates=candidates)

            tokenized_prompt = results["tokenized_inputs"]
            tokenized_candidates = results["tokenized_completions"]
            answers.append(results["answers"])

            for tokenized_completion in tokenized_candidates:
                tokenized_source = tokenized_prompt + tokenized_completion
                labels = [self.label_pad_token_id] * len(tokenized_prompt) + tokenized_completion

                model_inputs["input_ids"].append(tokenized_source)
                model_inputs["labels"].append(labels)
                model_inputs["attention_mask"].append([1 for _ in tokenized_source])

                model_inputs["prompt_input_ids"].append(tokenized_prompt)
                model_inputs["prompt_attention_mask"].append([1 for _ in tokenized_prompt])

                model_inputs["completion_input_ids"].append(tokenized_completion)
                model_inputs["completion_attention_mask"].append([1 for _ in tokenized_completion])

        self.pad_inputs(model_inputs)
        model_inputs = prepare_comp_attn_mask_llama(model_inputs, self.comp_args, self.comp_token,
                                                    self.sum_token, self.pad_token)

        model_inputs["answers"] = torch.as_tensor(answers)

        return model_inputs

