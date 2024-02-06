""" Huggingface LLaMA with Compressed Context Memory.
    The original code is created by Jang-Hyun Kim.
    GitHub Repository: https://github.com/snu-mllab/Context-Memory
"""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama import LlamaPreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaRotaryEmbedding,
    _make_causal_mask,
    _expand_mask,
    apply_rotary_pos_emb,
    rotate_half,
)
from transformers.utils import logging
from .generation_utils import GistGenerationMixin

from ..data.mask import get_comp_mask

logger = logging.get_logger(__name__)


class LinearMask(nn.Linear):

    def forward(self, input: Tensor, comp_mask=None) -> Tensor:
        return F.linear(input, self.weight, self.bias)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")
        self.q_proj = LinearMask(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = LinearMask(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = LinearMask(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = LinearMask(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                               max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        comp_mask: Optional[torch.Tensor] = None,
        sum_mask: Optional[torch.Tensor] = None,
        sum_attn_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        max_id: Optional[int] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        #####################################################
        ##### Modified: Feed-forward with conditional LoRA #####
        query_states = self.q_proj(hidden_states,
                                   comp_mask=comp_mask).view(bsz, q_len, self.num_heads,
                                                             self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states,
                                 comp_mask=comp_mask).view(bsz, q_len, self.num_heads,
                                                           self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states,
                                   comp_mask=comp_mask).view(bsz, q_len, self.num_heads,
                                                             self.head_dim).transpose(1, 2)
        #####################################################

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        #####################################################
        ##### Modified: Rotery embedding max length ######
        embed_len = kv_seq_len
        if max_id is not None:
            embed_len = max_id
        #####################################################

        cos, sin = self.rotary_emb(value_states, seq_len=embed_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin,
        #                                                 position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        #####################################################
        ##### Modified: Merge key values for sum token ######
        # (batch_size, seq_length, seq_length)
        if sum_attn_mask is not None:
            sum_attn_mask = sum_attn_mask.to(key_states.dtype)

            # (batch_size, n_heads, seq_length, dim_per_head)
            key_comp_avg = torch.matmul(sum_attn_mask.unsqueeze(1), key_states)
            value_comp_avg = torch.matmul(sum_attn_mask.unsqueeze(1), value_states)

            # (batch_size, 1, seq_length, 1)
            no_sum_mask = (1 - sum_mask).to(key_states.dtype).unsqueeze(1).unsqueeze(-1)

            key_states = no_sum_mask * key_states + key_comp_avg
            value_states = no_sum_mask * value_states + value_comp_avg
        #####################################################

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        ### Shift Pos: key pos is the pos in cache
        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        ###

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights,
                                     torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                             dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        #####################################################
        ##### Modified: Feed-forward with conditional LoRA ######
        attn_output = self.o_proj(attn_output, comp_mask=comp_mask)
        #####################################################

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        comp_mask: Optional[torch.Tensor] = None,
        sum_mask: Optional[torch.Tensor] = None,
        sum_attn_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        max_id: Optional[int] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            comp_mask=comp_mask,
            sum_mask=sum_mask,
            sum_attn_mask=sum_attn_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_id=max_id,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def update_position_ids(comp_mask, comp_token, attention_mask=None, type_="skip"):
    """Position ids for inputs with compression/sum tokens

    Args:
        comp_mask: [batch_size, seq_len]
        comp_token: compression token
        attention_mask: [batch_size, seq_len]
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(comp_mask)

    valid_position = attention_mask.to(comp_mask.dtype) * (1.0 - comp_mask)
    n_comp_tok = len(comp_token)

    if type_ == "skip":
        position_ids = valid_position.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # 1, 2, ..., n_comp_tok
        # Assume comp_token and sum_token has the same length
        position_ids_comp = (comp_mask.long().cumsum(-1) - 1) % n_comp_tok + 1
        position_ids = position_ids + (comp_mask * position_ids_comp).long()

    else:
        raise AssertionError(f"Unknown positional embedding type {type_}")

    return position_ids


class LlamaModelCCM(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each
    layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.comp_token = None
        self.sum_token = None
        self.comp_relative_embedding = config.comp_relative_embedding
        print("Compression token relative embedding: ", self.comp_relative_embedding)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from
    # transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds,
                                        past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask,
                                              inputs_embeds.dtype,
                                              tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else
                                       expanded_attn_mask + combined_attention_mask)

        return combined_attention_mask

    def get_comp_mask(self, input_ids):
        """
        Returns:
            comp_mask [batch_size, seq_len]: 1 for comp/sum token, 0 for others
            sum_mask [batch_size, seq_len]: 1 for sum token, 0 for others
            sum_attn_mask [batch_size, seq_len, seq_len]: A matrix for merging comp_tokens before sum_tokens
        """
        comp_mask = None
        if self.comp_token is not None:
            comp_mask = get_comp_mask(input_ids, self.comp_token).to(input_ids.device)

        sum_attn_mask = None
        sum_mask = None
        if self.sum_token is not None:
            # batchsize x seq_len
            sum_mask = get_comp_mask(input_ids, self.sum_token).to(input_ids.device)
            if sum_mask.sum() == 0:
                sum_mask = None

            if sum_mask is not None:
                assert comp_mask.sum() > 0, "sum_attn_mask is not None, but comp_mask is None"
                # batchsize x seq_len x seq_len
                batch_size, seq_len = input_ids.shape
                sum_attn_mask = torch.zeros((batch_size, seq_len, seq_len),
                                            device=input_ids.device).float()

                for k in range(len(self.comp_token)):
                    comp_loc_k = input_ids == self.comp_token[k]
                    sum_loc_k = input_ids == self.sum_token[k]
                    attn_k = sum_loc_k.unsqueeze(2) & comp_loc_k.unsqueeze(1)
                    sum_attn_mask += attn_k.float()

                sum_attn_mask = torch.tril(sum_attn_mask)

                sum_attn_mask = sum_attn_mask / torch.clamp(sum_attn_mask.sum(-1, keepdim=True),
                                                            min=0.1)

                # For conditional LoRA and positional embedding
                comp_mask = comp_mask + sum_mask

        return comp_mask, sum_mask, sum_attn_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_comp: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_id_offset: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        #####################################################
        ##### Modified: Get compression masks for conditional inference and memory update #####
        comp_mask, sum_mask, sum_attn_mask = self.get_comp_mask(input_ids)
        #####################################################

        output_attentions = (output_attentions
                             if output_attentions is not None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and "
                             "decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0
        seq_length_with_past = seq_length
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        assert attention_mask.shape[-1] == seq_length_with_past, "check attention mask size"

        #####################################################
        ##### Modified Part: Get Position_ids for inputs with compression/sum tokens #####
        if position_ids is None:
            if self.comp_relative_embedding == "base":
                # create position_ids on the fly for left padding
                # [batch_size, seq_length_with_past]
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids[:, past_key_values_length:]
            else:
                # Position ids with comp tokens
                ## Note: "update_position_ids" does not aware of past_key_values_length
                ## So you should properly set "pos_id_offset" when using past_key_values with compression
                ## (check inference.py test function for example)
                position_ids = update_position_ids(comp_mask,
                                                   self.comp_token,
                                                   attention_mask[:, -seq_length:],
                                                   type_=self.comp_relative_embedding)
            # consider prev comp tokens
            if pos_id_offset is not None:
                position_ids += pos_id_offset
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # for rotery embedding
        max_id = position_ids.max() + 1
        #####################################################

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        #####################################################
        ##### Modified: Get attention mask considering compression tokens ######
        if attention_mask_comp is not None:
            min_val = torch.tensor(torch.finfo(attention_mask.dtype).min)
            attention_mask_comp_float = torch.full_like(attention_mask, min_val)
            attention_mask_comp_float = attention_mask_comp_float.masked_fill(
                attention_mask_comp.bool(), 0.0)
            attention_mask = attention_mask + attention_mask_comp_float
        #####################################################

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. "
                                    "Setting `use_cache=False`...")
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (past_key_values[idx] if past_key_values is not None else None)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    comp_mask,
                    sum_mask,
                    sum_attn_mask,
                    attention_mask,
                    position_ids,
                    max_id,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    comp_mask=comp_mask,
                    sum_mask=sum_mask,
                    sum_attn_mask=sum_attn_mask,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    max_id=max_id,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                         if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM_CCM_Stream(LlamaPreTrainedModel, GistGenerationMixin):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelCCM(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.comp_token = None
        self.sum_token = None
        self.comp_relative_embedding = config.comp_relative_embedding

    def update_comp_token(self, comp_token, sum_token):
        self.comp_token = comp_token
        self.model.comp_token = comp_token

        self.sum_token = sum_token
        self.model.sum_token = sum_token

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_comp: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_id_offset: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will
                be ignored by default should you provide it.

                Indices can be obtained using [`AutoTokenizer`]. See
                [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size,
                    sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
                Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*,
                    returned when `use_cache=True` is passed or when
                    `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`,
                with each tuple having 2 tensors of shape `(batch_size,
                num_heads, sequence_length, embed_size_per_head)`) and 2
                additional tensors of shape `(batch_size, num_heads,
                encoder_sequence_length, embed_size_per_head)`. The two
                additional tensors are only required when the model is used as a
                decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the
                self-attention blocks and in the cross-attention blocks) that
                can be used (see `past_key_values` input) to speed up sequential
                decoding.

                If `past_key_values` are used, the user can optionally input
                only the last `decoder_input_ids` (those that don't have their
                past key value states given to this model) of shape
                `(batch_size, 1)` instead of all `decoder_input_ids` of shape
                `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size,
                    sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to
                directly pass an embedded representation.  This is useful if you
                want more control over how to convert `input_ids` indices into
                associated vectors than the model's internal embedding lookup
                matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`,
                    *optional*):
                Labels for computing the masked language modeling loss. Indices
                should either be in `[0, ..., config.vocab_size]` or -100 (see
                `input_ids` docstring). Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with
                labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are
                returned and can be used to speed up decoding (see
                `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention
                layers. See `attentions` under returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See
                `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a
                plain tuple.
            pos_id_offset ('torch.LongTensor', *optional*): 
                Offset for position_ids. This might required when using
                past_key_values with compression. 

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I
            can talk to you."
        ```"""
        output_attentions = (output_attentions
                             if output_attentions is not None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mask_comp=attention_mask_comp,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pos_id_offset=pos_id_offset,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        attention_mask_comp=None,
        inputs_embeds=None,
        pos_id_offset=None,
        first_time=False,
        **kwargs,
    ):
        # Don't trim the input ids if we are caching compression results
        # (indicated by non-None pos_id_offset).
        first_time_with_compression = first_time and pos_id_offset is not None

        #####################################################
        ##### Modified Part: Calculate position_ids for full inputs #####
        comp_mask = sum_mask = None
        if self.comp_token is not None:
            comp_mask = get_comp_mask(input_ids, self.comp_token).to(input_ids.device)
            comp_mask_all = comp_mask
            if self.sum_token is not None:
                sum_mask = get_comp_mask(input_ids, self.sum_token).to(input_ids.device)
                comp_mask_all = comp_mask + sum_mask

        input_len = input_ids.shape[1]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            if self.comp_relative_embedding == "base":
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = update_position_ids(comp_mask_all,
                                                   self.comp_token,
                                                   attention_mask[:, -input_len:],
                                                   type_=self.comp_relative_embedding)

            if pos_id_offset is not None:
                position_ids += pos_id_offset

        # Generation with caching
        if past_key_values and not first_time_with_compression:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        # Check size
        if (past_key_values is not None) and (attention_mask is not None):
            input_len = input_ids.shape[1]
            kv_len = past_key_values[0][0].shape[2]
            attn_len = attention_mask.shape[-1]
            t = f"kv: {kv_len}, input: {input_len}, attn: {attn_len}"
            assert (input_len == attn_len) or (kv_len + input_len == attn_len), t
        #####################################################

        # if `inputs_embeds` are passed, we only want to use them in the 1st
        # generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "attention_mask_comp": attention_mask_comp,
            "pos_id_offset": pos_id_offset,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
