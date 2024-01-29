from datasets import DatasetDict, load_dataset, concatenate_datasets
from .utils import nested_select, test_collator
from . import lamp, metaicl, dialogue
from .collator import DataCollator_LLAMA


def load_dataset_metric_collator(args, model, tokenizer):
    print("\n==== LOAD DATASET ====")

    pad_token = tokenizer.pad_token_id
    comp_token = tokenizer.comp_token_id
    sum_token = tokenizer.sum_token_id
    online = args.training.comp.comp_type == "online"

    if "merge" not in args.training.comp.attn_type:
        assert sum_token is None

    compute_metrics = metaicl.metrics.get_compute_metrics_fn(comp_token=comp_token,
                                                             tokenizer=tokenizer,
                                                             args=args)

    if args.data.dataset_name == "all":
        # Load datasets, MetaICL and dialogue datasets
        meta = metaicl.data.MetaICLData(
            tokenizer,
            args.is_llama,
            max_example_length=args.data.max_example_length,
            comp_token=comp_token,
            online=online,
            add_comp_token=args.training.comp.add_comp_token,
        )
        if args.data.max_eval_samples is not None:
            meta.eval_dataset = nested_select(meta.eval_dataset, args.data.max_eval_samples)
            meta.eval_dataset_option = nested_select(meta.eval_dataset_option,
                                                     args.data.max_eval_samples)

        dialog = dialogue.data_soda.DialogueDataset(
            tokenizer,
            comp_token=comp_token,
            online=online,
            add_comp_token=args.training.comp.add_comp_token,
        )

        # Concat datasets
        for key in meta.train_dataset.features.keys():
            if key not in dialog.train_dataset.features.keys():
                dialog.train_dataset = dialog.train_dataset.add_column(
                    key, [None] * len(dialog.train_dataset))

        for key in dialog.train_dataset.features.keys():
            if key not in meta.train_dataset.features.keys():
                meta.train_dataset = meta.train_dataset.add_column(key,
                                                                   [None] * len(meta.train_dataset))

        train_dataset = concatenate_datasets([meta.train_dataset, dialog.train_dataset])
        eval_dataset = {
            'rouge': meta.eval_dataset,
            "cls": meta.eval_dataset_option,
            "dialog": dialog.eval_dataset
        }

        # Load collator
        collator = DataCollator_LLAMA(
            meta=meta,
            dialog=dialog,
            tokenizer=tokenizer,
            comp_args=args.training.comp,
            model=model,
            max_length=args.data.max_length,
            k=args.data.k,
            comp_token=comp_token,
            sum_token=sum_token,
            pad_token=pad_token,
            seed_eval=args.data.seed_eval,
        )

        batch = [train_dataset[0], train_dataset[-1]]
        name = list(eval_dataset['rouge'].keys())[0]
        batch += [eval_dataset['rouge'][name][0]]
        name = list(eval_dataset['dialog'].keys())[0]
        batch += [eval_dataset['dialog'][name][0]]
        test_collator(collator, batch, tokenizer, args.is_llama)

    elif args.data.dataset_name == "metaicl":
        meta = metaicl.data.MetaICLData(
            tokenizer,
            args.is_llama,
            max_example_length=args.data.max_example_length,
            comp_token=comp_token,
            online=online,
            add_comp_token=args.training.comp.add_comp_token,
        )
        train_dataset = meta.train_dataset

        if args.data.max_eval_samples is not None:
            meta.eval_dataset = nested_select(meta.eval_dataset, args.data.max_eval_samples)
            meta.eval_dataset_option = nested_select(meta.eval_dataset_option,
                                                     args.data.max_eval_samples)
        eval_dataset = {'rouge': meta.eval_dataset, "cls": meta.eval_dataset_option}

        if args.is_llama:
            collator = metaicl.collator.DataCollatorForMetaICL_LLAMA(
                meta,
                tokenizer,
                comp_args=args.training.comp,
                model=model,
                max_length=args.data.max_length,
                k=args.data.k,
                comp_token=comp_token,
                sum_token=sum_token,
                pad_token=pad_token,
                seed_eval=args.data.seed_eval,
            )
        else:
            collator = metaicl.collator.DataCollatorForMetaICL_T5(
                meta,
                tokenizer,
                comp_args=args.training.comp,
                model=model,
                max_length=args.data.max_length,
                k=args.data.k,
                comp_token=comp_token,
                sum_token=sum_token,
                pad_token=pad_token,
                seed_eval=args.data.seed_eval,
            )

        batch = [train_dataset[0]]
        name = list(eval_dataset['rouge'].keys())[0]
        batch += [eval_dataset['rouge'][name][0]]
        test_collator(collator, batch, tokenizer, args.is_llama)

    elif args.data.dataset_name in ["dialog", "soda"]:
        if args.data.dataset_name == "dialog":
            data_fn = dialogue.data
        else:
            data_fn = dialogue.data_soda

        dialog = data_fn.DialogueDataset(
            tokenizer,
            comp_token=comp_token,
            online=online,
            add_comp_token=args.training.comp.add_comp_token,
        )
        train_dataset = dialog.train_dataset
        eval_dataset = dialog.eval_dataset

        if args.is_llama:
            collator = dialogue.collator.DataCollatorForDialogue_LLAMA(
                dialog,
                tokenizer,
                comp_args=args.training.comp,
                model=model,
                comp_token=comp_token,
                sum_token=sum_token,
                pad_token=pad_token,
            )
        else:
            raise AssertionError("dialogue only works with llama")

        name = list(eval_dataset.keys())[1]
        batch = [train_dataset[0], eval_dataset[name][0]]
        test_collator(collator, batch, tokenizer, args.is_llama)

    elif "lamp" in args.data.dataset_name:
        train_dataset, eval_dataset = load_dataset_from_args(args)

        collator = lamp.collator.LLaMACollatorForLaMP(
            tokenizer,
            lamp.retriever.BaseRetriever(k=args.data.k),
            comp_args=args.training.comp,
            model=model,
            padding="longest",
            max_source_length=args.training.max_source_length,
            max_target_length=args.training.max_target_length,
            pad_to_multiple_of=8 if args.training.fp16 else None,
            comp_token=tokenizer.comp_token_id,
            pad_token=tokenizer.pad_token_id,
            k=args.data.k,
        )
        compute_metrics = lamp.metrics.get_compute_metrics_fn(comp_token=comp_token,
                                                              tokenizer=tokenizer,
                                                              args=args)

        batch = [train_dataset[0]]
        batch += [eval_dataset['validation'][0]]
        print("TEST collator")
        test_collator(collator, batch, tokenizer, args.is_llama)

    else:
        raise NotImplementedError(f"Unknown dataset name {args.data.dataset_name}")

    return train_dataset, eval_dataset, compute_metrics, collator


def load_dataset_from_args(args):
    train_dataset = eval_dataset = None

    if "lamp" in args.data.dataset_name:
        lm_datasets = load_dataset(
            "src/data/lamp/lamp.py",
            name=args.data.dataset_name,
            cache_dir=args.model.cache_dir,
        )
        train_dataset = lm_datasets["train"]
        eval_dataset = DatasetDict({"validation": lm_datasets["validation"]})

    elif args.data.dataset_name == "metaicl":
        train_dataset = eval_dataset = None
    else:
        raise NotImplementedError(f"Unknown dataset name {args.data.dataset_name}")

    return train_dataset, eval_dataset
