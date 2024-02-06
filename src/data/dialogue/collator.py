import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union

from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy

from src.arguments import CompressionArguments
from ..collator import pad_inputs, prepare_comp_attn_mask_llama

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForDialogue_LLAMA:
    """Data collator for decoder-only models. Does left padding."""
    dialog: Optional[Any]
    tokenizer: PreTrainedTokenizerBase
    comp_args: CompressionArguments
    model: Optional[Any] = None
    comp_token: int = 32000
    sum_token: int = 32000
    return_tensors: str = "pt"
    padding: Union[bool, str, PaddingStrategy] = 'left'
    pad_token: int = 0
    label_pad_token_id: int = -100

    def __call__(self, batch, return_tensors=None):
        if type(self.comp_token) == int:
            self.comp_token = [self.comp_token]

        model_inputs = defaultdict(list)
        for instance in batch:
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
        sink_token = None
        if self.comp_args.sink:
            sink_token = self.tokenizer.bos_token_id

        model_inputs = pad_inputs("left", model_inputs, self.label_pad_token_id, self.pad_token)
        model_inputs = prepare_comp_attn_mask_llama(model_inputs,
                                                    self.comp_args,
                                                    self.comp_token,
                                                    self.sum_token,
                                                    self.pad_token,
                                                    sink_token=sink_token)

        return model_inputs
