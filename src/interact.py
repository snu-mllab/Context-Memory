""" The original code is created by Jang-Hyun Kim.
    GitHub Repository: https://github.com/snu-mllab/Context-Memory
"""

import hydra
import torch
from omegaconf.dictconfig import DictConfig
from transformers import GenerationConfig
from .arguments import Arguments, global_setup
from .model import load_model, load_pretrained
from .test import extract_comp_results, merge_comp_results, get_response, set_seperation_token


@torch.inference_mode()
def generate_and_compress(model, tokenizer, inputs, args):
    """Conduct generate and update compressed context memory
    Returns:
        generated output and compressed memory keys/values
    """
    assert model.comp_relative_embedding == "skip", "Only support 'skip' position_ids type for now"
    is_encoder_decoder = hasattr(model, "encoder")

    gen_kwargs = {}
    input_len = inputs["query"].shape[-1]
    kv_len = 0
    if inputs["past_key_values"]:
        kv_len = inputs["past_key_values"][0][0].shape[2]
    gen_kwargs["max_length"] = input_len + kv_len + 128
    gen_kwargs["num_beams"] = 1
    gen_kwargs["return_dict_in_generate"] = True

    if not is_encoder_decoder:
        gen_kwargs["generation_config"] = GenerationConfig(do_sample=False,
                                                           bos_token_id=1,
                                                           eos_token_id=2,
                                                           pad_token_id=0)

    # Generate response for given query
    generation_inputs = inputs["query"]
    gen_kwargs["past_key_values"] = inputs["past_key_values"]
    gen_kwargs["pos_id_offset"] = inputs["pos_id_offset"]
    gen_kwargs["attention_mask"] = generation_inputs.new_ones([1, kv_len + input_len])
    _ = gen_kwargs.pop("attention_mask_comp", None)

    outputs = model.base_model.generate(generation_inputs, **gen_kwargs)
    response = get_response(outputs, generation_inputs, tokenizer, is_encoder_decoder)

    # Compress
    cur_text = outputs.sequences[0]
    cur_text = cur_text[:-1]  # eos token
    ctx_len = cur_text.shape[-1]
    inputs["pos_id_offset"] += ctx_len

    comp_token_tensor = torch.tensor(tokenizer.comp_token_id, device='cuda').unsqueeze(0)
    n_tok = len(tokenizer.comp_token_id)

    past_key_values = outputs.past_key_values
    outputs = model.base_model.model(comp_token_tensor,
                                     past_key_values=past_key_values,
                                     use_cache=True,
                                     pos_id_offset=inputs["pos_id_offset"])

    if args.training.comp.attn_type == "concat_recur":
        comp_loc = [True] * n_tok * inputs["n_turn"] + [False] * ctx_len + [True] * n_tok
        comp_loc = torch.tensor(comp_loc, device='cuda')
        past_key_values = extract_comp_results(outputs.past_key_values, comp_loc)

    elif args.training.comp.attn_type == "merge_recur":
        comp_loc = [True] * n_tok * (inputs["n_turn"] > 0) + [False] * ctx_len + [True] * n_tok
        comp_loc = torch.tensor(comp_loc, device='cuda')
        past_key_values = extract_comp_results(outputs.past_key_values, comp_loc)
        if inputs["n_turn"] > 0:
            past_key_values = merge_comp_results(past_key_values, n_tok, inputs["n_turn"])

    return response, ctx_len, past_key_values


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)

    # Load model and tokenizer
    model, tokenizer = load_model(args)
    if args.training.eval_path != '':
        lora = args.training.peft
        load_pretrained(args.training.eval_path, model, lora=lora)
    model.eval()
    set_seperation_token(tokenizer, args.data.dataset_name)

    name = "CCM-concat" if "concat" in args.training.comp.attn_type else "CCM-merge"
    n_tok = len(tokenizer.comp_token_id)
    try:
        print()
        print("=" * 50)
        print(f"Interactive mode with {name}")
        print("Hi :) Write your prompt (Ctrl+c for break):")
        inputs = {}
        inputs["n_turn"] = 0
        inputs["past_key_values"] = None
        inputs["pos_id_offset"] = 0
        while True:
            query = input("\nUSER : ").strip()
            query = tokenizer.encode(query, add_special_tokens=False)
            if inputs["n_turn"] == 0:
                query = [tokenizer.bos_token_id] + query + tokenizer.sep_token_id_model
            else:
                query = tokenizer.sep_token_id_ + query + tokenizer.sep_token_id_model
            inputs["query"] = torch.tensor(query, device='cuda').unsqueeze(0)

            response, ctx_len, past_key_values = generate_and_compress(
                model, tokenizer, inputs, args)
            inputs["past_key_values"] = past_key_values
            inputs["n_turn"] += 1
            print(f"MODEL: {response}")
            print(f" => Compress {ctx_len} -> {n_tok} tokens"
                  f" (Memory key/value size = {past_key_values[0][0].shape[2]})")

    except KeyboardInterrupt:
        print("\rGood bye :)")


if __name__ == "__main__":
    main()
