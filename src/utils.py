"""Miscellaneous utils."""

import itertools
from typing import Any, List
import torch
from torch import nn


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
