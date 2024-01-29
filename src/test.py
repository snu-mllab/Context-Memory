""" The original code is created by Jang-Hyun Kim.
    GitHub Repository: https://github.com/snu-mllab/Context-Memory
"""

import os
import hydra
import torch
from omegaconf.dictconfig import DictConfig
from transformers import GenerationConfig
from .arguments import Arguments, global_setup
from .model import load_model, load_pretrained
from .data import mask


def extract_comp_results(past_key_values, loc):
    """Extract compressed memory keys/values from past_key_values
    KV shape: [n_layer, 2 (key/value)] x [bsz, n_head, seq_len, dim_per_head]
    """
    loc = loc.squeeze()
    n = len(loc)

    past_key_values_comp = []
    for n_layer in range(len(past_key_values)):
        kv = [past_key_values[n_layer][k][:, :, :n][:, :, loc] for k in range(2)]
        past_key_values_comp.append(kv)

    return past_key_values_comp


def merge_comp_results(past_key_values, n_tok, time_step):
    """Merge compressed memory keys/values
    KV shape: [n_layer, 2 (key/value)] x [bsz, n_head, n_comp_token x n_turn, dim_per_head]
    """
    assert past_key_values[0][0].shape[2] == 2 * n_tok

    past_key_values_comp = []
    for n_layer in range(len(past_key_values)):
        alpha = 1 / (time_step + 1)
        kv_l = past_key_values[n_layer]
        kv = [(1 - alpha) * kv_l[k][:, :, :n_tok] + alpha * kv_l[k][:, :, -n_tok:] for k in range(2)
             ]
        past_key_values_comp.append(kv)

    return past_key_values_comp


def get_response(outputs, generation_inputs, tokenizer, is_encoder_decoder):
    """ Decode generated outputs to string
    """
    generated_tokens = outputs.sequences
    if not is_encoder_decoder:
        # The prompt is included in the generated tokens. Remove this.
        assert (generated_tokens[:, :generation_inputs.shape[-1]] == generation_inputs).all()
        generated_tokens = generated_tokens[:, generation_inputs.shape[-1]:]

    if generated_tokens.shape[-1] > 1:
        generated_tokens = generated_tokens.squeeze()[:-1]  # Eos Token
        response = tokenizer.decode(generated_tokens)
    else:
        response = ""

    return response.strip()


def prepare_input(tokenizer, example, sum_recur=True, time_step=6):
    """Encode text sequences 
    Returns: 
        context_list: List of context token sequences
        context: Concatenated context tokens with <COMP> tokens
        query: Query tokens
        target: Target tokens
    """
    example_token = [tokenizer.encode(d, add_special_tokens=False) for d in example]

    comp_token = tokenizer.comp_token_id
    sum_token = tokenizer.sum_token_id
    sep_token = tokenizer.sep_token_id_

    n = min(len(example_token), time_step)
    context_list = example_token[:n - 2]
    context_list[0] = [tokenizer.bos_token_id] + context_list[0]

    context = []
    for c in context_list:
        context += c
        if comp_token is not None:
            context += comp_token
        if (sum_token is not None) and sum_recur:
            context += sum_token

        context += sep_token

    context = context[:-len(sep_token)]
    if (sum_token is not None) and not sum_recur:
        context += sum_token

    query = sep_token + example_token[n - 2] + sep_token
    target = example_token[n - 1]
    return context_list, context, query, target


def prepare_attn_mask_comp(tokenizer, input_ids, args):
    """ Make attention mask for inputs with compression tokens
    """
    comp_token = tokenizer.comp_token_id
    sum_token = tokenizer.sum_token_id

    if args.training.comp.attn_type == "concat_recur":
        comp_attn_fn = mask.get_comp_attn_mask_concat_recur
    elif args.training.comp.attn_type == "merge_recur":
        comp_attn_fn = mask.get_comp_attn_mask_recur
    else:
        raise NotImplementedError

    if args.training.comp.attn_type == "merge":
        attention_mask_comp = comp_attn_fn(input_ids, comp_token, sum_token=sum_token)
    elif args.training.comp.attn_type == "merge_recur":
        attention_mask_comp = comp_attn_fn(input_ids, sum_token)
    else:
        attention_mask_comp = comp_attn_fn(input_ids, comp_token)

    return attention_mask_comp


def set_seperation_token(tokenizer, dataset_name):
    if dataset_name == "pretrain":
        tokenizer.sep_token_id_ = tokenizer.encode('a\nA:',
                                                   add_special_tokens=False)[1:2]  # enter token
    elif dataset_name == "metaicl":
        tokenizer.sep_token_id_ = tokenizer.encode('a\nInput:', add_special_tokens=False)[1:]
    else:
        tokenizer.sep_token_id_ = tokenizer.encode('a\nA:', add_special_tokens=False)[1:]

    tokenizer.sep_token_id_model = tokenizer.encode('a\nOutput:', add_special_tokens=False)[1:]


@torch.inference_mode()
def test(model, tokenizer, inputs, args):
    """Test code for inference with compression
        * Check whether first three generation results are identical
    Returns:
        Generated: Single forward generation with text with multiple comp_tokens (compression occurs)
        Generated (recursive ver.): Recursive compression and then generate with compressed results
        W/o Context: Generation without context        
    """
    assert model.comp_relative_embedding == "skip", "Only support 'skip' position_ids type for now"
    is_encoder_decoder = hasattr(model, "encoder")

    gen_kwargs = {}
    context_length = inputs["context"].shape[-1] + inputs["query"].shape[-1]
    gen_kwargs["max_length"] = context_length + 128
    gen_kwargs["num_beams"] = 1
    gen_kwargs["return_dict_in_generate"] = True

    if not is_encoder_decoder:
        gen_kwargs["generation_config"] = GenerationConfig(do_sample=False,
                                                           bos_token_id=1,
                                                           eos_token_id=2,
                                                           pad_token_id=0)

    ###############################
    # Generation with compression mask (one step)
    generation_inputs = torch.cat([inputs["context"], inputs["query"]], dim=-1)
    gen_kwargs["attention_mask"] = torch.ones_like(generation_inputs)
    gen_kwargs["attention_mask_comp"] = prepare_attn_mask_comp(tokenizer, generation_inputs, args)

    outputs = model.base_model.generate(generation_inputs, **gen_kwargs)
    response = get_response(outputs, generation_inputs, tokenizer, is_encoder_decoder)
    print(f"{'Generated':28s}: {response}")

    ###############################
    # Generation with recursive compression
    past_key_values = None
    pos_id_offset = 0
    comp_token_tensor = torch.tensor(tokenizer.comp_token_id, device='cuda').unsqueeze(0)
    sep_token_tensor = torch.tensor(tokenizer.sep_token_id_, device='cuda').unsqueeze(0)
    n_tok = len(tokenizer.comp_token_id)
    for i, c in enumerate(inputs["context_list"]):
        if i > 0:
            c = torch.cat([sep_token_tensor, c], dim=-1)

        outputs = model.base_model.model(c,
                                         past_key_values=past_key_values,
                                         use_cache=True,
                                         pos_id_offset=pos_id_offset)
        pos_id_offset += c.shape[-1]

        past_key_values = outputs.past_key_values
        outputs = model.base_model.model(comp_token_tensor,
                                         past_key_values=past_key_values,
                                         use_cache=True,
                                         pos_id_offset=pos_id_offset)

        if args.training.comp.attn_type == "concat_recur":
            comp_loc = [True] * n_tok * i + [False] * c.shape[-1] + [True] * n_tok
            comp_loc = torch.tensor(comp_loc, device='cuda')
            past_key_values = extract_comp_results(outputs.past_key_values, comp_loc)

        elif args.training.comp.attn_type == "merge_recur":
            comp_loc = [True] * n_tok * (i > 0) + [False] * c.shape[-1] + [True] * n_tok
            comp_loc = torch.tensor(comp_loc, device='cuda')
            past_key_values = extract_comp_results(outputs.past_key_values, comp_loc)
            if i > 0:
                past_key_values = merge_comp_results(past_key_values, n_tok, i)

    gen_kwargs["past_key_values"] = past_key_values
    gen_kwargs["pos_id_offset"] = pos_id_offset

    generation_inputs = inputs["query"]
    input_len = generation_inputs.shape[-1]
    kv_len = gen_kwargs["past_key_values"][0][0].shape[2]
    gen_kwargs["attention_mask"] = generation_inputs.new_ones([1, kv_len + input_len])
    _ = gen_kwargs.pop("attention_mask_comp", None)

    outputs = model.base_model.generate(generation_inputs, **gen_kwargs)
    response = get_response(outputs, generation_inputs, tokenizer, is_encoder_decoder)
    print(f"{'Generated (recursive ver.)':28s}: {response}")

    ###############################
    # Generation without context
    generation_inputs = inputs["query"]
    generation_inputs[:, 0] = tokenizer.bos_token_id  # replace sep_token with bos_token
    gen_kwargs["attention_mask"] = torch.ones_like(generation_inputs)
    _ = gen_kwargs.pop("past_key_values", None)
    _ = gen_kwargs.pop("pos_id_offset", None)

    outputs = model.base_model.generate(generation_inputs, **gen_kwargs)
    response = get_response(outputs, generation_inputs, tokenizer,
                            is_encoder_decoder).split('\n')[0]
    print(f"{'W/o Context':28s}: {response}")

    return response, outputs.past_key_values


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)

    model, tokenizer = load_model(args)
    if args.training.eval_path != '':
        lora = args.training.peft
        load_pretrained(args.training.eval_path, model, lora=lora)
    model.eval()
    set_seperation_token(tokenizer, args.data.dataset_name)

    # Load sample data
    context_list, context, query, target = prepare_input(tokenizer, make_example())

    print(f'\n===========\n{tokenizer.decode(context + query).strip()}')
    print(f"{'Ground truth':28s}: {tokenizer.decode(target).strip()}")
    inputs = {}
    inputs["context_list"] = [torch.tensor(c, device='cuda').unsqueeze(0) for c in context_list]
    inputs["context"] = torch.tensor(context, device='cuda').unsqueeze(0)
    inputs["query"] = torch.tensor(query, device='cuda').unsqueeze(0)

    test(model, tokenizer, inputs, args)


def make_example():
    example = []
    example.append("Hi, I'm Janghyun. How are you?")
    example.append("I'm fine. It's great.")
    example.append("I'm fine, too. Where are you from?")
    example.append("I'm from Korea.")
    example.append("Oh, really? I'm from Korea, too. Do you remember my name? Please tell me.")
    example.append("Yes, I remember. Your name is Janghyun.")  # target output

    return example


if __name__ == "__main__":
    main()
