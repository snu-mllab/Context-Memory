""" The original code is created by Jang-Hyun Kim.
    GitHub Repository: https://github.com/snu-mllab/Context-Memory
"""

import os
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
)
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from .arch.config import PRETRAINED_VOCAB_SIZE_LLAMA, PRETRAINED_VOCAB_SIZE_T5
from .arch.gist_t5 import GistT5ForConditionalGeneration
from .arch.ccm_t5 import T5ForConditionalGeneration_CCM
from .arch.gist_llama import GistLlamaForCausalLM
from .arch.ccm_llama import LlamaForCausalLM_CCM

from .utils import SeparatedEmbedding, transpose
from . import peft_custom


def get_traininable_state_dict(model, state_dict, args):
    state_dict_ = get_peft_model_state_dict(model, state_dict)

    if args.training.comp.separate_embed:
        key = 'base_model.model.model.embed_tokens.comp_embeddings.weight'
        state_dict_[key] = state_dict[key]

    if args.training.finetune_embed:
        key = 'base_model.model.model.embed_tokens.weight'
        state_dict_[key] = state_dict[key]

    return state_dict_


def get_lora(args, model):
    config_path = None
    if args.training.eval_path != '':
        config_path = args.training.eval_path

    if config_path is None:
        # Initilzie LoRA
        TARGET_MODULES = args.training.target_modules.split(",")

        config = LoraConfig(
            r=args.training.lora_r,
            target_modules=TARGET_MODULES,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        config.save_pretrained(args.training.output_dir)

    else:
        config = LoraConfig().from_pretrained(config_path)

    print(f"LoRA with {config.target_modules}, r:{config.r}")

    embed = model.model.embed_tokens
    model = get_peft_model(model, config)

    if args.training.comp.separate_embed:
        embed.to(torch.float32)
        embed.comp_embeddings.weight.requires_grad = True

    model.print_trainable_parameters()
    return model


def get_conditional_lora(args, model):
    config_path = None
    if args.training.eval_path != '':
        config_path = args.training.eval_path

    if config_path is None:
        if args.is_t5:
            names = "q,k,v,o".split(",")
            names = [f"SelfAttention.{i}" for i in names]

            names_ = []
            for k in model.state_dict().keys():
                valid = False
                if "encoder" in k:
                    valid = [i in k for i in names]
                    valid = bool(sum(valid))
                if valid:
                    names_.append(k.split('.weight')[0])

            TARGET_MODULES = names_
        else:
            TARGET_MODULES = args.training.target_modules.split(",")

        config = LoraConfig(
            r=args.training.lora_r,
            target_modules=TARGET_MODULES,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        config.save_pretrained(args.training.output_dir)
    else:
        config = LoraConfig().from_pretrained(config_path)

    if args.is_t5:
        embed = model.shared
    else:
        embed = model.model.embed_tokens

    print(f"Conditional LoRA with {config.target_modules}, r:{config.r}")
    model = peft_custom.get_peft_model(model, config)

    if args.training.comp.separate_embed:
        embed.to(torch.float32)
        embed.comp_embeddings.weight.requires_grad = True

    model.print_trainable_parameters()
    return model


def load_lora_weight(path, model, merge=False):
    # Embedding can be included
    lora_weight_saved = torch.load(os.path.join(path, 'pytorch_model.bin'))

    if merge:
        config = LoraConfig().from_pretrained(path)
        scaling = config.lora_alpha / config.r

        for name, param in model.named_parameters():
            dtype_orig = param.data.dtype

            lora_name = 'base_model.model.' + name.split('.weight')[0]
            if f'{lora_name}.lora_A.weight' in lora_weight_saved:
                dtype_lora = lora_weight_saved[f'{lora_name}.lora_B.weight'].dtype

                param.data = param.data.to(dtype_lora) + (transpose(
                    lora_weight_saved[f'{lora_name}.lora_B.weight']
                    @ lora_weight_saved[f'{lora_name}.lora_A.weight'],
                    config.fan_in_fan_out,
                ) * scaling)

                param.data = param.data.to(dtype_orig)

        print(f"Merge LoRA weight from {path}")

    else:
        lora_weight = {}

        target_key = list(model.state_dict().keys())
        for k, v in lora_weight_saved.items():
            # backward compatibility
            if 'gist_embeddings' in k:
                k = k.replace('gist_embeddings', 'comp_embeddings')
            if k not in target_key:
                k = k[:-7] + '.default.weight'

            lora_weight[k] = v

        for k in lora_weight:
            if k not in target_key:
                raise AssertionError(f"Key {k} not in model")

        key = 'base_model.model.model.embed_tokens.comp_embeddings.weight'
        if key in lora_weight:
            model_size = model.state_dict()[key].shape[0]
            load_size = lora_weight[key].shape[0]

            if model_size != load_size:
                min_len = min(model_size, load_size)

                embed = model.model.model.embed_tokens.comp_embeddings.weight
                embed.data[:min_len] = lora_weight[key][:min_len]

                lora_weight.pop(key)

        model.load_state_dict(lora_weight, strict=False)
        print(f"Load LoRA weight from {path}")


def load_pretrained(path, model, lora=False, merge=False):
    if lora:
        load_lora_weight(path, model, merge=merge)
    else:
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'))
        model.load_state_dict(state_dict)
        print(f"Load model from {path}")

        del state_dict


def load_model(args):
    print("\n==== LOAD MODEL, TOKENIZER ====")
    model_path = args.model.model_name_or_path

    ## Config
    config_kwargs = {
        "cache_dir": args.model.cache_dir,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }
    if args.model.config_name:
        config = AutoConfig.from_pretrained(args.model.config_name, **config_kwargs)
    elif model_path:
        config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    else:
        raise ValueError("Check model_name!")

    ## Tokenizer
    args.is_t5 = any(t in model_path.lower() for t in ("t5", "tk"))
    args.is_llama = any(t in model_path.lower() for t in ("llama",))
    if args.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer_name,
                                                  use_fast=args.model.use_fast_tokenizer,
                                                  **config_kwargs)
    elif model_path:
        if args.is_llama:
            tokenizer = LlamaTokenizer.from_pretrained(model_path,
                                                       use_fast=args.model.use_fast_tokenizer,
                                                       **config_kwargs)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                      use_fast=args.model.use_fast_tokenizer,
                                                      **config_kwargs)
    else:
        raise ValueError("check tokenizer_name or model_name_or_path")

    # Positional encoding type for <COMP> tokens
    config.comp_relative_embedding = args.training.comp.relative_embedding
    config.attn_type = args.training.comp.attn_type
    print(f"Attention type: {config.attn_type}")

    ## Model
    if args.is_t5:
        model_cls = GistT5ForConditionalGeneration
        if args.training.comp.cond_lora:
            model_cls = T5ForConditionalGeneration_CCM
            print("Use conditional LoRA!")
    elif args.is_llama:
        model_cls = GistLlamaForCausalLM
        if args.training.comp.cond_lora:
            model_cls = LlamaForCausalLM_CCM
            print("Use conditional LoRA!")
    else:
        raise ValueError(f"Model type {model_path} not supported")

    if args.model.pretrained:
        dtype = torch.float16 if args.training.peft else torch.float32
        model = model_cls.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in model_path),
            config=config,
            torch_dtype=dtype,
            device_map='auto',
            **config_kwargs,
        )
        print(f"Load {model_path}")

        if args.is_llama:
            model.config.pad_token_id = tokenizer.pad_token_id = 0
            model.config.bos_token_id = tokenizer.bos_token_id = 1
            model.config.eos_token_id = tokenizer.eos_token_id = 2
    else:
        model = model_cls(config)

    # Load base model
    if args.training.load_path != '':
        load_pretrained(args.training.load_path, model, lora=args.training.peft, merge=True)

    # Update comp tokens and LoRA for compression
    tokenizer.comp_token_id = tokenizer.sum_token_id = None
    if args.training.comp.add_comp_token:
        update_compression_token(args, model, tokenizer)

    if args.training.peft and not args.training.comp.cond_lora:
        model = get_lora(args, model)
    elif args.training.comp.cond_lora:
        model = get_conditional_lora(args, model)

    print("pad, bos, eos, comp token id: ", tokenizer.pad_token_id, tokenizer.bos_token_id,
          tokenizer.eos_token_id, tokenizer.comp_token_id)
    return model, tokenizer


def update_compression_token(args, model, tokenizer):
    n_tok = args.training.comp.num_comp_tokens
    if "merge" in args.training.comp.attn_type:
        n_tok *= 2  # For sum tokens which have merged comp token keys/values

    # Check if COMP token is already initialized
    if args.is_t5 and len(tokenizer) == PRETRAINED_VOCAB_SIZE_T5 + n_tok:
        assert model.shared.weight.shape[0] == PRETRAINED_VOCAB_SIZE_T5 + n_tok

    elif args.is_llama and len(tokenizer) == PRETRAINED_VOCAB_SIZE_LLAMA + n_tok:
        assert (model.model.embed_tokens.weight.shape[0] == PRETRAINED_VOCAB_SIZE_LLAMA + n_tok)
        assert model.lm_head.weight.shape[0] == PRETRAINED_VOCAB_SIZE_LLAMA + n_tok

    # Initialize COMP token
    else:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"<COMP{k}>" for k in range(n_tok)]})

        if args.training.comp.separate_embed:
            model.resize_token_embeddings(len(tokenizer) - n_tok)

            if args.is_t5:
                model.shared = SeparatedEmbedding(model.shared, n_tok)
                model.encoder.embed_tokens = model.shared
                model.decoder.embed_tokens = model.shared
                embed = model.shared

            elif args.is_llama:
                model.model.embed_tokens = SeparatedEmbedding(model.model.embed_tokens, n_tok)
                embed = model.model.embed_tokens

                # embed.to(torch.float32)
                embed.comp_embeddings.to(torch.float32)

            for p in model.parameters():
                p.requires_grad_(False)

            embed.comp_embeddings.weight.requires_grad = True
            print("Separate embedding for compression tokens!", embed)

        else:
            model.resize_token_embeddings(len(tokenizer))
            # Set new word embedding to average of existing word embeddings. For why,
            # see https://nlp.stanford.edu/~johnhew/vocab-expansion.html
            with torch.no_grad():
                if args.is_t5:
                    embed = model.shared
                elif args.is_llama:
                    embed = model.model.embed_tokens

                # fixed initialization for COMP embedding
                for k in range(n_tok):
                    embed.weight[-k - 1] = embed.weight[:-n_tok][k::n_tok].mean(0)

                if args.is_llama:
                    for k in range(n_tok):
                        model.lm_head.weight[-k -
                                             1] = model.lm_head.weight[:-n_tok][k::n_tok].mean(0)

                    if args.training.peft:
                        # If we train embedding, we need to cast to float32
                        if args.training.finetune_embed:
                            embed.to(torch.float32)
                        else:
                            embed.weight.requires_grad = False

                        model.lm_head.weight.requires_grad = False

    if "merge" in args.training.comp.attn_type:
        tokenizer.comp_token_id = tokenizer.additional_special_tokens_ids[:n_tok // 2]
        tokenizer.sum_token_id = tokenizer.additional_special_tokens_ids[n_tok // 2:]
    else:
        tokenizer.comp_token_id = tokenizer.additional_special_tokens_ids

    if args.is_t5:
        model.encoder.comp_token = tokenizer.comp_token_id
        model.encoder.sum_token = tokenizer.sum_token_id
    else:
        model.update_comp_token(tokenizer.comp_token_id, tokenizer.sum_token_id)

    print("comp tokens (w/ sum tokens): ", ''.join(tokenizer.additional_special_tokens))
