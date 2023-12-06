import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy

from . import mask
from src.arguments import CompressionArguments

logger = logging.getLogger(__name__)


def pad_inputs(loc, model_inputs, label_pad_token_id, pad_token):
    # Pad inputs, convert to tensor.
    for key, value in model_inputs.items():
        if key in ["is_answer", "idx"]:
            continue

        if "labels" in key:
            pad_token_id = label_pad_token_id
        elif "attention_mask" in key:
            pad_token_id = 0
        else:
            pad_token_id = pad_token

        if loc == "left":
            # To left-pad inputs, reverse, then right-pad, then reverse.
            # Pad attention too (with pad_token_id)
            if type(value[0]) == torch.Tensor:
                value_tensors = [torch.flip(v, dims=(0, )) for v in value]
            else:
                value_tensors = [torch.tensor(v[::-1]) for v in value]
            model_inputs[key] = torch.fliplr(
                pad_sequence(
                    value_tensors,
                    batch_first=True,
                    padding_value=pad_token_id,
                ))

        elif loc == "right":
            # Pad attention too (with pad_token_id)
            if type(value[0]) == torch.Tensor:
                value_tensors = value
            else:
                value_tensors = [torch.tensor(v) for v in value]

            model_inputs[key] = pad_sequence(value_tensors,
                                             batch_first=True,
                                             padding_value=pad_token_id)
    return model_inputs


def prepare_comp_attn_mask_llama(model_inputs, comp_args, comp_token, sum_token, pad_token):
    # Construct compression attention mask
    attn_type = comp_args.attn_type
    if comp_args.comp_type in ["pos_control", "neg_control"] or not comp_args.add_comp_token:
        comp_attn_fn = mask.get_pos_attn_mask

    elif comp_args.comp_type == "fixed":
        if attn_type == "gist":
            comp_attn_fn = mask.get_comp_attn_mask_recur
        else:
            raise NotImplementedError(f"Unknown attention type {attn_type}")

    elif comp_args.comp_type == "online":
        if attn_type == "concat":
            comp_attn_fn = mask.get_comp_attn_mask_concat
        elif attn_type == "merge":
            comp_attn_fn = mask.get_comp_attn_mask_merge
        elif attn_type == "concat_recur":
            comp_attn_fn = mask.get_comp_attn_mask_concat_recur
        elif attn_type == "merge_recur":
            comp_attn_fn = mask.get_comp_attn_mask_recur
        else:
            raise NotImplementedError(f"Unknown attention type {attn_type}")

    else:
        raise ValueError(f"Unknown compression type {comp_args.comp_type}")

    if attn_type == "merge":
        model_inputs["attention_mask_comp"] = comp_attn_fn(
            model_inputs["input_ids"],
            comp_token,
            sum_token=sum_token,
            pad_token=pad_token,
        )
        model_inputs["prompt_attention_mask_comp"] = comp_attn_fn(
            model_inputs["prompt_input_ids"],
            comp_token,
            sum_token=sum_token,
            pad_token=pad_token,
        )
    elif attn_type == "merge_recur":
        model_inputs["attention_mask_comp"] = comp_attn_fn(
            model_inputs["input_ids"],
            sum_token,
            pad_token=pad_token,
        )
        model_inputs["prompt_attention_mask_comp"] = comp_attn_fn(
            model_inputs["prompt_input_ids"],
            sum_token,
            pad_token=pad_token,
        )
    else:
        model_inputs["attention_mask_comp"] = comp_attn_fn(
            model_inputs["input_ids"],
            comp_token,
            pad_token=pad_token,
        )
        model_inputs["prompt_attention_mask_comp"] = comp_attn_fn(
            model_inputs["prompt_input_ids"],
            comp_token,
            pad_token=pad_token,
        )

    return model_inputs


@dataclass
class DataCollator_LLAMA:
    """Data collator for decoder-only models. Does left padding."""

    meta: Optional[Any]
    dialog: Optional[Any]
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
            task = "metaicl"
            if "dialog" in instance:
                if instance["dialog"] is not None:
                    task = "dialog"

            if task == "metaicl":
                # For classification evaluation
                if "is_answer" in instance:
                    model_inputs["is_answer"].append(instance["is_answer"])
                    model_inputs["idx"].append(instance["idx"])

                instance = self.meta.add_demonstration(
                    instance,
                    max_length=self.max_length,
                    k=self.k,
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

            else:
                instance = self.dialog.sample_dialog(
                    instance,
                    random_k=True,
                    sum_token=self.sum_token,
                    sum_recur=self.comp_args.attn_type == "merge_recur",
                    neg_control=self.comp_args.comp_type == "neg_control",
                )

                input_token = instance['input_ids']
                output_token = instance['output_ids']

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
