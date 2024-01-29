"""Misc data utils."""

from datasets import DatasetDict


def strip_special_tokens(s):
    s = s.replace("<pad> ", "").replace("</s>", "").replace("<pad>", "")
    s = s.replace("<unk>", "").replace("⁇", "").strip()
    return s


def nested_select(datasets: DatasetDict, max_len: int, **kwargs):
    return DatasetDict(
        {k: v.select(range(min(max_len, len(v))), **kwargs) for k, v in datasets.items()})


def test_collator(collator, batch, tokenizer, is_llama):
    model_inputs = collator(batch)

    if is_llama:
        for key in model_inputs.keys():
            try:
                print(key, model_inputs[key].shape)
            except:
                print(key)

        for i, instance in enumerate(model_inputs["input_ids"]):
            print("\n* Full input: ")
            print(tokenizer.decode(instance))

            print("\n* Original sample: ")
            mask = model_inputs["attention_mask"][i]
            print(tokenizer.decode([t for k, t in enumerate(instance) if mask[k]]))

            print("\n* Compressed sample: ")
            mask = model_inputs["attention_mask_comp"][i, 0][-1]
            print(tokenizer.decode([t for i, t in enumerate(instance) if mask[i]]))

            print("\n* Label: ")
            labels = model_inputs["labels"][i]
            print(tokenizer.decode([t for t in labels if t != -100]))
    else:
        for key in model_inputs.keys():
            try:
                print(key, model_inputs[key].shape)
            except:
                print(key)

        for i, instance in enumerate(model_inputs["input_ids"]):
            print("\n* Full input: ")
            print(tokenizer.decode(instance))

            print("\n* Original sample: ")
            mask = model_inputs["attention_mask"][i]
            if len(mask.shape) == 2:
                mask = mask[-1]
            print(tokenizer.decode([t for k, t in enumerate(instance) if mask[k]]))

            print("\n* Compressed sample: ")
            mask = model_inputs["cross_attention_mask"][i]
            if len(mask.shape) == 2:
                mask = mask[-1]
            print(tokenizer.decode([t for k, t in enumerate(instance) if mask[k]]))

            print("\n* Label: ")
            labels = model_inputs["labels"][i]
            print(tokenizer.decode([t for t in labels if t != -100]))
