from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from transformers.models.bert.modeling_bert import BertSelfAttention
from chop.models.patched.llama.modeling_llama import LlamaSdpaAttention
from transformers.models.vit.modeling_vit import ViTSelfAttention

from chop.models.patched.llama.configuration_llama import LlamaConfig

from chop.nn.quantized.modules.linear import (
    LinearInteger,
)
from chop.nn.quantized.functional import fixed_softermax
from chop.nn.quantized.functional import matmul_integer

from typing import Optional, Tuple, Union


class _BertSelfAttentionBase(BertSelfAttention):
    def __init__(
        self,
        config,
        q_config: dict = None,
        out_q_config: dict = None,
        position_embedding_type=None,
        bias=True,
        output_tensor_only=False,
    ) -> None:
        super().__init__(config, position_embedding_type)
        self.bypass = False
        self.q_config = q_config
        self.out_q_config = out_q_config
        self.bias = bias
        self.output_tensor_only = output_tensor_only

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        out = super().forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        if self.output_tensor_only:
            return out[0]
        return out


class _LlamaSdpaAttentionBase(LlamaSdpaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        q_config: dict = None,
        out_q_config: dict = None,
        output_tensor_only=False,
    ):
        super().__init__(config, layer_idx)
        self.bypass = False
        self.q_config = q_config
        self.out_q_config = out_q_config
        self.output_tensor_only = output_tensor_only

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        out = super().forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        )
        if self.output_tensor_only:
            return out[0]
        return out

class _ViTSelfAttentionBase(ViTSelfAttention):
    def __init__(
        self,
        config,
        q_config: dict = None,
        out_q_config: dict = None,
        bias=True,
        output_tensor_only=False,
    ) -> None:
        super().__init__(config)
        self.bypass = False
        self.q_config = q_config
        self.out_q_config = out_q_config
        self.bias = bias
        self.output_tensor_only = output_tensor_only

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        out = super().forward(
            hidden_states,
            head_mask,
            output_attentions,
        )
        if self.output_tensor_only:
            return out[0]
        return out


class BertSelfAttentionInteger(_BertSelfAttentionBase):
    def __init__(
        self,
        config,
        q_config: dict = None,
        out_q_config: dict = None,
        position_embedding_type=None,
        bias=True,
        floor=False,
        output_tensor_only=False,
    ) -> None:
        super().__init__(
            config,
            q_config,
            out_q_config,
            position_embedding_type,
            bias=bias,
            output_tensor_only=output_tensor_only,
        )
        self.query = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=q_config,
            out_config=out_q_config,
            bias=bias,
            floor=floor,
        )
        self.key = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=q_config,
            out_config=out_q_config,
            bias=bias,
            floor=floor,
        )
        self.value = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=q_config,
            out_config=out_q_config,
            bias=bias,
            floor=floor,
        )
        # * Matmul is used for Q @ K^T and Scores @ V where the input values have already
        # * been casted to the output precision, so we provide the output precision to the
        # * software model
        self.matmul = partial(
            matmul_integer,
            config={
                "data_in_width": self.q_config["data_in_width"],
                "data_in_frac_width": self.q_config["data_in_frac_width"],
                "weight_width": self.q_config["weight_width"],
                "weight_frac_width": self.q_config["weight_frac_width"],
            },
            out_config=out_q_config,
            floor=floor,
        )


class LlamaSdpaAttentionInteger(_LlamaSdpaAttentionBase):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        q_config: dict = None,
        out_q_config: dict = None,
        output_tensor_only=False,
    ):
        super().__init__(
            config,
            layer_idx,
            q_config,
            out_q_config,
            output_tensor_only=output_tensor_only,
        )
        self.q_proj = LinearInteger(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
            config=q_config,
            out_config=out_q_config,
        )
        self.k_proj = LinearInteger(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            config=q_config,
            out_config=out_q_config,
        )
        self.v_proj = LinearInteger(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            config=q_config,
            out_config=out_q_config,
        )
        self.o_proj = LinearInteger(
            self.hidden_size,
            self.hidden_size,
            bias=config.attention_bias,
            config=q_config,
            out_config=out_q_config,
        )


class ViTSelfAttentionInteger(_ViTSelfAttentionBase):
    def __init__(
        self,
        config,
        q_config: dict = None,
        out_q_config: dict = None,
        bias=True,
        floor=False,
        output_tensor_only=False,
    ) -> None:
        super().__init__(
            config,
            q_config,
            out_q_config,
            bias=bias,
            output_tensor_only=output_tensor_only,
        )
        self.query = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=q_config,
            out_config=out_q_config,
            bias=bias,
            floor=floor,
        )
        self.key = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=q_config,
            out_config=out_q_config,
            bias=bias,
            floor=floor,
        )
        self.value = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=q_config,
            out_config=out_q_config,
            bias=bias,
            floor=floor,
        )
        # * Matmul is used for Q @ K^T and Scores @ V where the input values have already
        # * been casted to the output precision, so we provide the output precision to the
        # * software model
        self.matmul = partial(
            matmul_integer,
            config={
                "data_in_width": self.q_config["data_in_width"],
                "data_in_frac_width": self.q_config["data_in_frac_width"],
                "weight_width": self.q_config["weight_width"],
                "weight_frac_width": self.q_config["weight_frac_width"],
            },
            out_config=out_q_config,
            floor=floor,
        )
