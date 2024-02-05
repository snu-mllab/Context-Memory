# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training script, adapted from huggingface's run_clm.py example.
"""

import logging
import os
import sys

import hydra
import torch
from omegaconf.dictconfig import DictConfig
from transformers import (
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from .arguments import Arguments, global_setup
from .callbacks import CustomWandbCallback, EvaluateFirstStepCallback
from .trainer_seq2seq import CompSeq2SeqTrainer
from .model import load_model, get_traininable_state_dict, load_pretrained
from .data.load import load_dataset_metric_collator
from .utils import check_model

# Will error if the minimal version of Transformers is not installed.
check_min_version("4.28.0.dev0")
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


def detect_last_checkpoint(args):
    # Detecting last checkpoint.
    last_checkpoint = None
    if (os.path.isdir(args.training.output_dir) and args.training.do_train and
            not args.training.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(args.training.output_dir)
        if last_checkpoint is None and len(os.listdir(args.training.output_dir)) > 0:
            existing_files = os.listdir(args.training.output_dir)
            logger.warning(
                ("Output directory (%s) already exists and "
                 "is not empty. Existing files: %s. "
                 "Training anyways as these may just be output files."),
                args.training.output_dir,
                str(existing_files),
            )
        elif (last_checkpoint is not None and args.training.resume_from_checkpoint is None):
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                        "avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from "
                        "scratch.")
    return last_checkpoint


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)
    last_checkpoint = None
    # last_checkpoint = detect_last_checkpoint(args)  # Load the existing checkpoints
    set_seed(args.training.seed)
    JID = int(os.environ.get("SLURM_JOB_ID", 0))
    if JID > 0:
        print(f"slurm job id: {JID}")

    # ==== LOAD MODEL ====
    model, tokenizer = load_model(args)

    # Load evaluation model
    if args.training.eval_path != '':
        lora = args.training.peft
        load_pretrained(args.training.eval_path, model, lora=lora)

    if args.training.peft:
        old_state_dict = model.state_dict
        model.state_dict = (lambda self, *_, **__: get_traininable_state_dict(
            self, old_state_dict(), args)).__get__(model, type(model))

    # ==== LOAD DATASET and COLLATOR ====
    train_dataset, eval_dataset, compute_metrics, data_collator = load_dataset_metric_collator(
        args, model, tokenizer)
    if args.data.dataset_name in ["metaicl", "unified"]:
        eval_dataset_total = eval_dataset
        eval_dataset = eval_dataset_total["rouge"]
        eval_dataset_cls = eval_dataset_total["cls"]
        if args.data.dataset_name == "unified":
            eval_dataset_dialog = eval_dataset_total["dialog"]

    # ==== INITIALIZE TRAINER ====
    custom_callbacks = []
    if args.wandb.log:
        custom_callbacks.append(CustomWandbCallback(args))
    if args.training.evaluate_before_train:
        custom_callbacks.append(EvaluateFirstStepCallback())

    trainer = CompSeq2SeqTrainer(
        model=model,
        args=args.training,
        train_dataset=train_dataset if args.training.do_train else None,
        eval_dataset=dict(eval_dataset) if args.training.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if args.training.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=None,
        callbacks=custom_callbacks,
    )

    # ==== START TRAINING and EVALUATION ====
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    check_model(model, peft=args.training.peft)
    if args.training.do_train:
        checkpoint = None
        if args.training.resume_from_checkpoint is not None:
            checkpoint = args.training.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (args.data.max_train_samples
                             if args.data.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.training.do_eval:  # always run evaluation
        all_eval_metrics = {}

        if "lamp" in args.data.dataset_name:
            lamp_index = int(args.data.dataset_name[4])
            do_classification = lamp_index in [1, 2, 3]
            model = model.eval()
            metrics = trainer.evaluate_lamp_cls(
                eval_dataset["validation"],
                lamp_index=2,
            )
            all_eval_metrics.update(metrics)

        elif (args.data.dataset_name == "metaicl") and eval_dataset_cls is not None:
            for eval_name, to_eval in eval_dataset_cls.items():
                logger.info(f"*** Evaluate {eval_name} ***")

                # Do Classification evaluation
                metrics = trainer.evaluate_metaicl(to_eval)
                metrics = {f"{eval_name}_{k}": v for k, v in metrics.items()}
                all_eval_metrics.update(metrics)

                # Do Generation (ROUGE) evaluation
                if not args.training.do_train and args.training.eval_rouge:
                    metrics = trainer.evaluate(eval_dataset[eval_name])
                    update_eval_metrics(metrics, all_eval_metrics, to_eval, eval_name, args)

        # The default trainer conduct evaluation
        elif args.data.dataset_name in ["dialog", "soda"]:
            for eval_name, to_eval in eval_dataset.items():
                logger.info(f"*** Evaluate {eval_name} ***")

                # Do perplexity evaluation
                metrics = trainer.evaluate_perp(to_eval)
                metrics = {f"{eval_name}_{k}": v for k, v in metrics.items()}
                all_eval_metrics.update(metrics)

                print(all_eval_metrics)
                # Do Generation (ROUGE) evaluation
                if not args.training.do_train and args.training.eval_rouge:
                    metrics = trainer.evaluate(to_eval)
                    update_eval_metrics(metrics, all_eval_metrics, to_eval, eval_name, args)

        elif args.data.dataset_name == "unified":
            # Eval dialog
            for eval_name, to_eval in eval_dataset_dialog.items():
                logger.info(f"*** Evaluate {eval_name} ***")

                # Do perplexity evaluation
                metrics = trainer.evaluate_perp(to_eval)
                metrics = {f"{eval_name}_{k}": v for k, v in metrics.items()}
                all_eval_metrics.update(metrics)

            # Eval metaicl
            for eval_name, to_eval in eval_dataset_cls.items():
                logger.info(f"*** Evaluate {eval_name} ***")

                # Do Classification evaluation
                metrics = trainer.evaluate_metaicl(to_eval)
                metrics = {f"{eval_name}_{k}": v for k, v in metrics.items()}
                all_eval_metrics.update(metrics)

                # Do Generation (ROUGE) evaluation
                if not args.training.do_train and args.training.eval_rouge:
                    metrics = trainer.evaluate(eval_dataset[eval_name])
                    update_eval_metrics(metrics, all_eval_metrics, to_eval, eval_name, args)

        if len(all_eval_metrics) > 0:
            trainer.log_metrics("eval", all_eval_metrics)
            trainer.save_metrics("eval", all_eval_metrics)


def update_eval_metrics(metrics, all_eval_metrics, to_eval, eval_name, args):
    max_eval_samples = (args.data.max_eval_samples
                        if args.data.max_eval_samples is not None else len(to_eval))
    metrics["eval_samples"] = min(max_eval_samples, len(to_eval))
    metrics = {(f"{eval_name}_{k}" if k != "epoch" else k): v for k, v in metrics.items()}
    all_eval_metrics.update(metrics)


if __name__ == "__main__":
    main()
