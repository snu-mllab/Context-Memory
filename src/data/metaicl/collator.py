import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy

from .. import mask
from .data import MetaICLData
from src.arguments import CompressionArguments
from ..collator import pad_inputs, prepare_comp_attn_mask_llama

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForMetaICL_T5:
    meta: MetaICLData
    tokenizer: PreTrainedTokenizerBase
    comp_args: CompressionArguments
    model: Optional[Any] = None
    max_length: Optional[int] = None
    comp_token: int = 32100
    sum_token: int = 32000
    return_tensors: str = "pt"
    k: int = 16
    padding: Union[bool, str, PaddingStrategy] = 'right'
    pad_token: int = 0
    label_pad_token_id: int = -100
    seed_eval: int = 100

    def __call__(self, batch, return_tensors=None):
        if type(self.comp_token) == int:
            self.comp_token = [self.comp_token]

        model_inputs = defaultdict(list)
        for instance in batch:
            # For evaluation
            if "is_answer" in instance:
                model_inputs["is_answer"].append(instance["is_answer"])
                model_inputs["idx"].append(instance["idx"])
                model_inputs["baseline"].append(self.meta.output_prefix)

            instance = self.meta.add_demonstration(instance,
                                                   max_length=self.max_length,
                                                   k=self.k,
                                                   random_k=self.comp_args.random_k,
                                                   seed_eval=self.seed_eval,
                                                   sum_token=self.sum_token)

            input_token = instance['input_ids']
            output_token = instance['output_ids']

            model_inputs["input_ids"].append(input_token)
            model_inputs["labels"].append(output_token)
            model_inputs["attention_mask"].append([1 for _ in input_token])

        # Right padding
        model_inputs = pad_inputs("right", model_inputs, self.label_pad_token_id, self.pad_token)

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids

        positive_mask = model_inputs["attention_mask"]
        # modify attention mask
        if self.comp_args.comp_type == "pos_control" or not self.comp_args.add_comp_token:
            # Don't change anything, just set cross attention mask.
            model_inputs["cross_attention_mask"] = positive_mask

        elif self.comp_args.comp_type == "fixed" and self.comp_args.attn_type == "gist":
            model_inputs["attention_mask"] = mask.get_comp_attn_mask_recur(
                model_inputs["input_ids"],
                self.comp_token,
                pad_token=self.pad_token,
            ).squeeze(1)  # Squeeze dim 1 as T5 codebase expects 3D mask.
            # Decoder cross attn cannot see prior to the last gist token.
            model_inputs["cross_attention_mask"] = mask.cross_attention_comp_last(
                model_inputs["input_ids"],
                self.comp_token,
                pad_token=self.pad_token,
            )

        elif self.comp_args.attn_type == "concat":
            model_inputs["attention_mask"] = mask.get_comp_attn_mask_concat(
                model_inputs["input_ids"],
                self.comp_token,
                pad_token=self.pad_token,
            ).squeeze(1)  # Squeeze dim 1 as T5 codebase expects 3D mask.
            # Decoder cross attn cannot see prior to the last gist token.
            model_inputs["cross_attention_mask"] = mask.cross_attention_comp_concat(
                model_inputs["input_ids"],
                self.comp_token,
                pad_token=self.pad_token,
            )

        elif self.comp_args.attn_type == "merge":
            model_inputs["attention_mask"] = mask.get_comp_attn_mask_merge(
                model_inputs["input_ids"],
                self.comp_token,
                sum_token=self.sum_token,
                pad_token=self.pad_token,
            ).squeeze(1)  # Squeeze dim 1 as T5 codebase expects 3D mask.
            # Decoder cross attn cannot see prior to the last gist token.
            model_inputs["cross_attention_mask"] = mask.cross_attention_comp_last(
                model_inputs["input_ids"],
                self.sum_token,
                pad_token=self.pad_token,
            )

        else:
            raise ValueError(
                f"Invalid comp_type: {self.comp_args.comp_type}, {self.comp_args.attn_type}")

        return model_inputs


@dataclass
class DataCollatorForMetaICL_LLAMA:
    """Data collator for decoder-only models. Does left padding."""

    meta: MetaICLData
    tokenizer: PreTrainedTokenizerBase
    comp_args: CompressionArguments
    model: Optional[Any] = None
    max_length: Optional[int] = 1024
    comp_token: int = 32000
    sum_token: int = 32000
    return_tensors: str = "pt"
    k: int = 16
    padding: Union[bool, str, PaddingStrategy] = 'left'
    pad_token: int = 0
    label_pad_token_id: int = -100
    seed_eval: int = 100

    def __call__(self, batch, return_tensors=None):
        if type(self.comp_token) == int:
            self.comp_token = [self.comp_token]

        model_inputs = defaultdict(list)
        for instance in batch:
            # For evaluation
            if "is_answer" in instance:
                model_inputs["is_answer"].append(instance["is_answer"])
                model_inputs["idx"].append(instance["idx"])

            instance = self.meta.add_demonstration(
                instance,
                max_length=self.max_length,
                k=self.k,
                random_k=self.comp_args.random_k,
                seed_eval=self.seed_eval,
                sum_token=self.sum_token,
                sum_recur=self.comp_args.attn_type == "merge_recur",
            )

            input_token = instance['input_ids']
            output_token = instance['output_ids']

            if "is_answer" in model_inputs:
                baseline = self.meta.output_prefix + output_token
                baseline_labels = [self.label_pad_token_id] * len(
                    self.meta.output_prefix) + output_token

                model_inputs["baseline"].append(baseline)
                model_inputs["baseline_labels"].append(baseline_labels)
                model_inputs["baseline_attention_mask"].append([1 for _ in baseline])

            full_token = input_token + output_token
            labels = [self.label_pad_token_id] * len(input_token) + output_token

            model_inputs["input_ids"].append(full_token)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1 for _ in full_token])

            model_inputs["prompt_input_ids"].append(input_token)
            model_inputs["prompt_attention_mask"].append([1 for _ in input_token])

            model_inputs["completion_input_ids"].append(output_token)
            model_inputs["completion_attention_mask"].append([1 for _ in output_token])

        # Left padding
        model_inputs = pad_inputs("left", model_inputs, self.label_pad_token_id, self.pad_token)
        model_inputs = prepare_comp_attn_mask_llama(model_inputs, self.comp_args, self.comp_token,
                                                    self.sum_token, self.pad_token)

        return model_inputs
