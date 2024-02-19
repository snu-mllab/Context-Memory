"""Miscellaneous utils."""

import itertools
from typing import Any, List
import torch
from torch import nn
from .data import mask


def first_mismatch(a: List[Any], b: List[Any], window: int = 10):
    """Returns first mismatch as well as sublists for debugging."""
    for i, (x, y) in enumerate(itertools.zip_longest(a, b)):
        if x != y:
            window_slice = slice(i - window, i + window)
            return (x, y), (a[window_slice], b[window_slice])
    return None


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def check_model(model, peft=False):
    mem = torch.cuda.memory_allocated() / 10**6
    print(f"\nCheck model: {model.dtype}, {model.device} (current Mem {mem:.0f} MB)")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            break
    if peft:
        model.print_trainable_parameters()
    else:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {pytorch_total_params:,}")


class SeparatedEmbedding(nn.Module):
    """Separate embedding for tokens."""

    def __init__(self, embedding, n_new_tokens):
        super().__init__()
        n_vocab, n_dim = embedding.weight.shape

        self.n_vocab = n_vocab
        self.n_new = n_new_tokens
        self.embeddings = embedding
        self.padding_idx = embedding.padding_idx
        self.device = self.embeddings.weight.device

        self.comp_embeddings = nn.Embedding(n_new_tokens, n_dim).to(self.device)
        self._initialize_comp_token_embedding()

    def _initialize_comp_token_embedding(self):
        with torch.no_grad():
            for k in range(self.n_new):
                self.comp_embeddings.weight[-k - 1] = self.embeddings.weight[k::self.n_new].mean(0)

    def forward(self, input_ids):
        mask = input_ids >= self.n_vocab  # comp tokens

        input_ids_ = input_ids.clone()
        input_ids_[mask] = 0  # mask out comp tokens
        input_embeds = self.embeddings(input_ids_)

        input_ids_ = input_ids.clone()
        input_ids_ -= self.n_vocab
        input_ids_[~mask] = 0  # mask out original tokens
        comp_token_embeds = self.comp_embeddings(input_ids_)

        mask = mask.to(comp_token_embeds.dtype).unsqueeze(-1)
        embeds = mask * comp_token_embeds + (1 - mask) * input_embeds  # mix

        return embeds


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
    assert past_key_values[0][0].shape[2] in [2 * n_tok, 2 * n_tok + 1]
    if past_key_values[0][0].shape[2] == 2 * n_tok + 1:
        offset = 1

    past_key_values_comp = []
    for n_layer in range(len(past_key_values)):
        alpha = 1 / (time_step + 1)
        kv_l = past_key_values[n_layer]
        kv = [(1 - alpha) * kv_l[k][:, :, offset:n_tok + offset] + alpha * kv_l[k][:, :, -n_tok:]
              for k in range(2)]
        if offset > 0:
            for k in range(2):
                kv[k] = torch.cat([kv_l[k][:, :, :offset], kv[k]], dim=2)

        past_key_values_comp.append(kv)

    return past_key_values_comp


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

    sink_token = None
    if args.training.comp.sink:
        sink_token = tokenizer.bos_token_id

    if args.training.comp.attn_type == "merge":
        attention_mask_comp = comp_attn_fn(input_ids,
                                           comp_token,
                                           sum_token=sum_token,
                                           sink_token=sink_token)
    elif args.training.comp.attn_type == "merge_recur":
        attention_mask_comp = comp_attn_fn(input_ids, sum_token, sink_token=sink_token)
    else:
        attention_mask_comp = comp_attn_fn(input_ids, comp_token, sink_token=sink_token)

    return attention_mask_comp


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
