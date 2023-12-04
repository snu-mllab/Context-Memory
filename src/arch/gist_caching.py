"""
Utilities for caching gist tokens.
Code adapted from Gisting GitHub repo: https://github.com/jayelm/gisting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput

from ..data.mask import get_gist_index


@dataclass
class CompressedKV:
    last_hidden_state: torch.FloatTensor
    past_key_values: Tuple[Tuple[torch.FloatTensor]]
    # In case needed, e.g. for absolute position embeddings.
    comp_indices: Optional[torch.LongTensor] = None

    @classmethod
    def from_model_outputs(
        cls,
        model_outputs: ModelOutput,
        input_ids: torch.LongTensor,
        comp_token: int,
        merge=False,
        time_step=0,
    ) -> CompressedKV:
        """
        Computes gist activations for the model.

        Returns:
            CompressedKV object containing the kv activations for the gist
            tokens of the encoder, and the start position of the position ids.
        """
        assert hasattr(model_outputs, "last_hidden_state")
        assert hasattr(model_outputs, "past_key_values")

        num_comp_tokens = len(comp_token)

        past_key_values = model_outputs.past_key_values
        batch_size, num_heads, seq_length, hidden_size = past_key_values[0][0].shape
        device = past_key_values[0][0].device

        def kv():
            return torch.zeros(
                (batch_size, num_heads, num_comp_tokens, hidden_size),
                dtype=past_key_values[0][0].dtype,
                device=device,
            )

        # Save key/values and hidden states corresponding to gist tokens only.
        last_hidden_state = torch.zeros(
            (batch_size, num_comp_tokens, model_outputs.last_hidden_state.shape[-1]),
            dtype=model_outputs.last_hidden_state.dtype,
            device=device,
        )
        # Same structure as past_key_values, but only has `num_comp_tokens` spots.
        gist_past_key_values = [(kv(), kv()) for _ in past_key_values]
        gist_starts = []
        for batch_i, input_row in enumerate(input_ids):
            # Find start and end gist position.
            gist_start, gist_end = get_gist_index(input_row, comp_token, raise_if_no_tokens=True)

            # For each hidden layer, save the activations corresponding to the
            # gist token.
            for layer_i, (past_k, past_v) in enumerate(past_key_values):
                gist_past_key_values[layer_i][0][batch_i] = past_k[batch_i, :, gist_start:gist_end]
                gist_past_key_values[layer_i][1][batch_i] = past_v[batch_i, :, gist_start:gist_end]

            # Keep track of gist starts for the batch.
            gist_starts.append(gist_start)

            # Save last hidden state corresponding to gist token.
            last_hidden_state[batch_i] = model_outputs.last_hidden_state[batch_i,
                                                                         gist_start:gist_end]

        return CompressedKV(
            last_hidden_state=last_hidden_state,
            past_key_values=tuple(gist_past_key_values),
            comp_indices=torch.tensor(gist_starts, dtype=torch.int64, device=device),
        )
