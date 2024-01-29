"""
Compressed Context Memory version of Seq2SeqTrainer.
Code adapted from Gisting GitHub repo: https://github.com/jayelm/gisting
"""

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import Dataset
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, Seq2SeqTrainer
from transformers.debug_utils import DebugOption
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
from transformers.utils import is_torch_tpu_available, logging
from .data.lamp.utils import classification_candidates

logger = logging.get_logger(__name__)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm  # type: ignore
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl  # type: ignore


class CompSeq2SeqTrainer(Seq2SeqTrainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model.
                Most models expect the targets under the argument `labels`.
                Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor],
            Optional[torch.Tensor]]: A tuple with the loss, logits and labels
            (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = self._gen_kwargs.copy()
        if (gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None):
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (gen_kwargs["num_beams"] if gen_kwargs.get("num_beams")
                                   is not None else self.model.config.num_beams)
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus")
                                     is not None else default_synced_gpus)

        is_encoder_decoder = hasattr(self.model, "encoder")
        original_inputs = None
        if not is_encoder_decoder and "prompt_input_ids" in inputs:
            # Decoder-only models: when generating, we don't want to generate
            # with the full text, but with the prompt text.
            original_inputs = {
                k: inputs[k] for k in ["input_ids", "attention_mask", "attention_mask_comp"]
            }
            inputs["input_ids"] = inputs["prompt_input_ids"]
            inputs["attention_mask"] = inputs["prompt_attention_mask"]
            inputs["attention_mask_comp"] = inputs["prompt_attention_mask_comp"]

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # ==== BEGIN CHANGES ====
        if "attention_mask_comp" in inputs:
            gen_kwargs["attention_mask_comp"] = inputs.get("attention_mask_comp", None)
        if "cross_attention_mask" in inputs:
            gen_kwargs["cross_attention_mask"] = inputs.get("cross_attention_mask", None)
        if "past_key_values" in inputs:
            gen_kwargs["past_key_values"] = inputs.get("past_key_values", None)
        if "pos_id_offset" in inputs:
            gen_kwargs["pos_id_offset"] = inputs.get("pos_id_offset", None)
        # ==== END CHANGES ====

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if (is_encoder_decoder and
                self.model.encoder.main_input_name != self.model.main_input_name):
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        if not is_encoder_decoder:
            # logger.warning("Overwriting existing generation config due to "
            #    "DeepSpeed bug. If model is not LLAMA, check this.")
            gen_kwargs["generation_config"] = GenerationConfig(
                do_sample=False,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )

        if self.args.peft or self.args.comp.cond_lora:
            generated_tokens = self.model.base_model.generate(
                generation_inputs,
                **gen_kwargs,
            )
        else:
            generated_tokens = self.model.generate(
                generation_inputs,
                **gen_kwargs,
            )

        if not is_encoder_decoder:
            # The prompt is included in the generated tokens. Remove this.
            assert (generated_tokens[:, :generation_inputs.shape[-1]] == generation_inputs).all()
            generated_tokens = generated_tokens[:, generation_inputs.shape[-1]:]

        # in case the batch is shorter than max length, the output should be padded
        if (gen_kwargs.get("max_length") is not None and
                generated_tokens.shape[-1] < gen_kwargs["max_length"]):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens,
                                                            gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens,
                                                            gen_kwargs["max_new_tokens"] + 1)

        # Replace original inputs when computing standard LM loss.
        if not is_encoder_decoder and original_inputs is not None:
            inputs.update(original_inputs)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = (self.label_smoother(outputs, inputs["labels"]).mean().detach())
                else:
                    loss = ((outputs["loss"]
                             if isinstance(outputs, dict) else outputs[0]).mean().detach())
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if (gen_kwargs.get("max_length") is not None and
                    labels.shape[-1] < gen_kwargs["max_length"]):
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                    gen_kwargs["max_new_tokens"] + 1):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        resume_from_checkpoint: Optional[str] = None,
        lamp_index: Optional[int] = None,
        do_classification: bool = False,
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute
        metrics, as they are task-dependent (pass it to the init
        `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If
                it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must
                implement the `__len__` method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a
                dictionary) that should be ignored when gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For
                example the metrics "bleu" will be named "eval_bleu" if the
                prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential
            metrics computed from the predictions. The dictionary also contains
            the epoch number which comes from the training state.
        """
        if hasattr(self.data_collator, "retriever"):
            self.data_collator.retriever.reset_eval_seed()
        gen_kwargs = gen_kwargs.copy()
        if (gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None):
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (gen_kwargs["num_beams"] if gen_kwargs.get("num_beams")
                                   is not None else self.args.generation_num_beams)
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (self.prediction_loop
                     if self.args.use_legacy_prediction_loop else self.evaluation_loop)
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics,
            # otherwise we defer to self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            ))

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile,
            # execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control,
                                                         output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        if do_classification:
            classification_metrics = self.evaluate_classification(eval_dataset, lamp_index)
            classification_metrics = {
                f"{metric_key_prefix}_{k}": v for k, v in classification_metrics.items()
            }
            output.metrics.update(classification_metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        resume_from_checkpoint: Optional[str] = None,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and
        `Trainer.predict()`.  Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (prediction_loss_only
                                if prediction_loss_only is not None else args.prediction_loss_only)

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should
            # be able to do eval from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self,
                num_training_steps=1_000_000,
                resume_from_checkpoint=resume_from_checkpoint,
                inference=resume_from_checkpoint is None,
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or
        # ``predict`` isn't called while ``train`` is running, cast it to the
        # right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader
                # in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model,
                                                        inputs,
                                                        prediction_loss_only,
                                                        ignore_keys=ignore_keys)
            # For logging generation results
            inputs_decode = None
            if args.include_inputs_for_metrics:
                if "prompt_input_ids" in inputs:
                    inputs_decode = self._prepare_input(inputs["prompt_input_ids"])
                else:
                    inputs_decode = self._prepare_input(inputs["input_ids"])

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = (losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0))
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = (labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100))
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (inputs_decode if inputs_host is None else nested_concat(
                    inputs_host, inputs_decode, padding_index=-100))
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = (logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100))
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done
            # enough accumulation steps.
            if (args.eval_accumulation_steps is not None and
                (step + 1) % args.eval_accumulation_steps == 0):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0))
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100))
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (inputs_decode if all_inputs is None else nested_concat(
                        all_inputs, inputs_decode, padding_index=-100))
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (labels if all_labels is None else nested_concat(
                        all_labels, labels, padding_index=-100))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0))
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100))
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (inputs_decode if all_inputs is None else nested_concat(
                all_inputs, inputs_decode, padding_index=-100))
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100))

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type,
        # but whether the dataset has the right methods. Therefore we need to
        # make sure it also has the attribute.
        elif (isinstance(eval_dataset, IterableDatasetShard) and
              getattr(eval_dataset, "num_examples", 0) > 0):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a
        # distributed training, the number of samplers has been rounded to a
        # multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if (self.compute_metrics is not None and all_preds is not None and all_labels is not None):
            if args.write_outputs:
                os.makedirs(os.path.join(args.output_dir, "outputs"), exist_ok=True)
                output_file = os.path.join(
                    args.output_dir,
                    f"outputs/{self.state.global_step}-{eval_dataset._split}.csv",
                )
            else:
                output_file = None
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs),
                    output_file=output_file,
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels),
                    output_file=output_file,
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )

    @torch.inference_mode()
    def _loglikelihood_seq2seq(self, batch, check=False, split_batch=-1):
        """
        Source code adapted and modified from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/models/huggingface.py#L638
        Compute log likelihood log p(y | x) with Encoder-Decoder Language Model (Such as T5)
        """
        batch = {k: v.to(self.args.device) for k, v in batch.items()}

        if check:
            print(self.tokenizer.batch_decode(batch['input_ids']))

        if split_batch > 0:

            len_batch = len(batch["labels"])
            assert len_batch % split_batch == 0

            logits = []
            batch_split_size = len_batch // split_batch

            for i in range(0, len_batch, batch_split_size):
                batch_i = {k: v[i:i + batch_split_size] for k, v in batch.items()}
                output_i = self.model(**batch_i)
                logits_i = output_i.logits
                logits_i = logits_i.to(dtype=torch.float)
                logits.append(logits_i)
            logits = torch.cat(logits, dim=0)
            log_probs = F.log_softmax(logits, dim=-1)
        else:
            # Compute logits
            output = self.model(**batch)
            logits = output.logits  # [B, maxlen, vocab]
            logits = logits.to(dtype=torch.float)
            log_probs = F.log_softmax(logits, dim=-1)

        labels = batch["labels"]

        # Compute log-likelihood
        log_likelihoods = np.zeros(len(labels))
        log_likelihoods_normalized = np.zeros(len(labels))
        label_length = np.zeros(len(labels))

        eos = self.tokenizer.eos_token_id
        pad_label = -100
        for i in range(len(labels)):
            y_valid = torch.logical_and(labels[i] != eos,
                                        labels[i] != pad_label)  # ignore padding and eos
            y_tokens = labels[i].unsqueeze(dim=-1)
            y_log_probs = log_probs[i]
            if check:
                print(self.tokenizer.decode(y_tokens[y_valid].squeeze()))

            y_log_probs = torch.gather(y_log_probs[y_valid], 1, y_tokens[y_valid]).squeeze(dim=-1)

            log_likelihoods[i] = y_log_probs.sum().item()
            log_likelihoods_normalized[i] = y_log_probs.mean().item()
            label_length[i] = y_valid.sum().item()

        ret = {
            "loglikelihoods": log_likelihoods,
            "loglikelihoods_normalized": log_likelihoods_normalized,
            "label_length": label_length,
        }

        return ret

    @torch.inference_mode()
    def _loglikelihood_clm(self, batch, check=False, split_batch=-1):
        """
        Source code adapted and modified from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/base.py#L271
        Compute log likelihood log p(y | x) with Causal Language Model (Such as LLaMA)
        """
        batch = {k: v.to(self.args.device) for k, v in batch.items()}

        if check:
            print(self.tokenizer.batch_decode(batch['input_ids']))

        if split_batch > 0:

            len_batch = len(batch["labels"])
            assert len_batch % split_batch == 0

            # log_probs = []
            logits = []

            batch_split_size = len_batch // split_batch

            for i in range(0, len_batch, batch_split_size):
                batch_i = {k: v[i:i + batch_split_size] for k, v in batch.items()}
                output_i = self.model(**batch_i)
                logits_i = output_i.logits
                logits_i = logits_i.to(dtype=torch.float)
                logits.append(logits_i)
            logits = torch.cat(logits, dim=0)
            log_probs = F.log_softmax(logits, dim=-1)
        else:
            # Compute logits
            output = self.model(**batch)
            logits = output.logits  # [B, maxlen, vocab]
            logits = logits.to(dtype=torch.float)
            log_probs = F.log_softmax(logits, dim=-1)

        # output_token
        labels = batch["labels"]

        # Compute log-likelihood
        log_likelihoods = np.zeros(len(labels))
        log_likelihoods_normalized = np.zeros(len(labels))
        label_length = np.zeros(len(labels))

        eos = self.tokenizer.eos_token_id
        pad_label = -100
        for i in range(len(labels)):
            y_tokens = labels[i].unsqueeze(dim=-1)
            y_log_probs = log_probs[i]

            y_tokens_shift = y_tokens[1:, :]
            y_log_probs_shift = y_log_probs[:-1, :]

            y_valid = torch.logical_and(y_tokens_shift != eos,
                                        y_tokens_shift != pad_label)  # ignore padding and eos
            y_valid = y_valid.squeeze(dim=-1)
            if check:
                print(self.tokenizer.decode(y_tokens_shift[y_valid].squeeze()))

            y_log_probs = torch.gather(y_log_probs_shift[y_valid], 1,
                                       y_tokens_shift[y_valid]).squeeze(dim=-1)

            log_likelihoods[i] = y_log_probs.sum().item()
            log_likelihoods_normalized[i] = y_log_probs.mean().item()
            label_length[i] = y_valid.sum().item()

        ret = {
            "loglikelihoods": log_likelihoods,
            "loglikelihoods_normalized": log_likelihoods_normalized,
            "label_length": label_length,
        }

        return ret

    def evaluate_lamp_cls(self, dataset, lamp_index):
        is_t5 = hasattr(self.model, "encoder")

        if self.model.training:
            self.model.eval()

        assert not self.model.training
        self.data_collator.retriever.reset_eval_seed()

        collate_fn = self.data_collator.collate_for_classification
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_size=self.args.eval_batch_size)

        likelihood_fn = self._loglikelihood_seq2seq if is_t5 else self._loglikelihood_clm

        candidates_batch = self.data_collator.get_candidate_batch(lamp_index)

        base_likelihood = likelihood_fn(candidates_batch)["loglikelihoods"]
        base_likelihood = np.expand_dims(base_likelihood, axis=0)

        prediction_list = []
        prediction_base_list = []
        prediction_norm_list = []
        answer_list = []

        logger.info(f"***** Evaluating LaMP{lamp_index} classification *****")
        for batch in tqdm(dataloader):
            answers = batch.pop("answers")
            answers = answers.numpy()
            input_ids = batch["input_ids"]

            batch_size = len(answers)
            assert len(input_ids) % batch_size == 0
            num_class = len(input_ids) // batch_size

            split_batch = len(classification_candidates[lamp_index])

            results = likelihood_fn(batch, split_batch=split_batch)

            loglikelihoods = results["loglikelihoods"]
            loglikelihoods = loglikelihoods.reshape(batch_size, num_class)
            preds = loglikelihoods.argmax(axis=-1)
            preds_base = (loglikelihoods - base_likelihood).argmax(axis=-1)

            loglikelihoods_norm = results["loglikelihoods_normalized"]
            loglikelihoods_norm = loglikelihoods_norm.reshape(batch_size, num_class)
            preds_norm = loglikelihoods_norm.argmax(axis=-1)

            prediction_list += preds.tolist()
            prediction_norm_list += preds_norm.tolist()
            prediction_base_list += preds_base.tolist()
            answer_list += answers.tolist()

        total = {}
        preds = np.array(prediction_list)
        preds_norm = np.array(prediction_norm_list)
        preds_base = np.array(prediction_base_list)
        answers = np.array(answer_list)

        acc = (preds == answers).mean()
        acc_base = (preds_base == answers).mean()
        acc_norm = (preds_norm == answers).mean()

        total = {
            "eval_acc": acc,
            # we do not report these
            # "eval_acc_base": acc_base,
            # "eval_acc_norm": acc_norm,
        }

        for k, v in total.items():
            total[k] = round(v * 100, 2)

        return total

    def evaluate_metaicl(self, dataset):
        if not self.is_in_train:
            if self.args.fp16_full_eval:
                self.model = self.model.to(dtype=torch.float16, device=self.args.device)
            elif self.args.bf16_full_eval:
                self.model = self.model.to(dtype=torch.bfloat16, device=self.args.device)
        self.model.eval()

        is_t5 = hasattr(self.model, "encoder")
        dataloader = DataLoader(dataset,
                                collate_fn=self.data_collator,
                                batch_size=self.args.eval_batch_size)

        likelihood_fn = self._loglikelihood_seq2seq if is_t5 else self._loglikelihood_clm

        mle_total = []
        mle_norm_total = []
        mle_base_total = []
        is_answer_total = []
        idx_total = []

        for batch in tqdm(dataloader):
            is_answer_total += batch.pop("is_answer")
            idx_total += batch.pop("idx")

            batch_ = {}
            if is_t5:
                # For T5 identical input_ids for baseline (e.g., Output:)
                # Thus we do not need to feed baseline_attention_mask
                batch_['input_ids'] = batch['baseline']
                batch_['labels'] = batch['labels']
                batch_['decoder_input_ids'] = batch['decoder_input_ids']
                batch.pop("baseline")
            else:
                batch_['input_ids'] = batch['baseline']
                batch_['labels'] = batch['baseline_labels']
                batch_['attention_mask'] = batch['baseline_attention_mask']
                batch.pop("baseline")
                batch.pop("baseline_labels")
                batch.pop("baseline_attention_mask")

            results_ = likelihood_fn(batch_)
            mle_base_total.append(results_["loglikelihoods"])

            results = likelihood_fn(batch)

            loglikelihoods = results["loglikelihoods"]
            mle_total.append(loglikelihoods)

            loglikelihoods_norm = results["loglikelihoods_normalized"]
            mle_norm_total.append(loglikelihoods_norm)

        is_answer_total = np.array(is_answer_total)
        idx_total = np.array(idx_total)
        mle_total = np.concatenate(mle_total)
        mle_norm_total = np.concatenate(mle_norm_total)
        mle_base_total = np.concatenate(mle_base_total)
        mle_base_total = mle_total - mle_base_total  # Baseline subtracted MLE

        pred_list = []
        pred_norm_list = []
        pred_base_list = []
        answer_list = []

        for i in range(np.max(idx_total)):
            idx = idx_total == i

            mle = mle_total[idx]
            mle_norm = mle_norm_total[idx]
            mle_base = mle_base_total[idx]
            is_answer = is_answer_total[idx]

            pred_list.append(np.argmax(mle))
            pred_norm_list.append(np.argmax(mle_norm))
            pred_base_list.append(np.argmax(mle_base))
            answer_list.append(np.argmax(is_answer))

        accuracy = np.mean(np.array(pred_list) == np.array(answer_list))
        accuracy_norm = np.mean(np.array(pred_norm_list) == np.array(answer_list))
        accuracy_base = np.mean(np.array(pred_base_list) == np.array(answer_list))

        total = {
            "eval_acc": accuracy * 100,
            "eval_acc_norm": accuracy_norm * 100,
            "eval_acc_base": accuracy_base * 100,
        }
        return total

    def evaluate_perp(self, dataset):
        if not self.is_in_train:
            if self.args.fp16_full_eval:
                self.model = self.model.to(dtype=torch.float16, device=self.args.device)
            elif self.args.bf16_full_eval:
                self.model = self.model.to(dtype=torch.bfloat16, device=self.args.device)
        self.model.eval()

        is_t5 = hasattr(self.model, "encoder")
        dataloader = DataLoader(dataset,
                                collate_fn=self.data_collator,
                                batch_size=self.args.eval_batch_size)

        likelihood_fn = self._loglikelihood_seq2seq if is_t5 else self._loglikelihood_clm

        loss_sentence = []
        perp_sentence = []
        token_total = []
        loss_sum = []

        for batch in tqdm(dataloader):
            results = likelihood_fn(batch)

            entropy_sum = -results["loglikelihoods"]
            loss_sum.append(entropy_sum)
            token_total.append(results["label_length"])

            entropy = -results["loglikelihoods_normalized"]
            loss_sentence.append(entropy)
            perp_sentence.append(np.exp(entropy))

        loss_sum = np.concatenate(loss_sum)
        token_total = np.concatenate(token_total)

        loss_sentence = np.concatenate(loss_sentence)
        perp_sentence = np.concatenate(perp_sentence)

        loss = np.mean(loss_sentence)
        loss_token = np.sum(loss_sum) / np.sum(token_total)
        perp_sentence = np.mean(perp_sentence)

        total = {
            "loss": loss,
            "perplexity": np.exp(loss_token),
            # "perplexity_sentence": perp_sentence,  # We do not report this
        }
        return total
