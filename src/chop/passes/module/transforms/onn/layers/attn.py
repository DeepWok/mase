import logging
import math
from collections import namedtuple
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mase_triton.optical_compute import OpticalTransformerFunctions as OTFunctions
from mase_triton.optical_compute import layers as OTLayers
from mase_triton.optical_compute.layers import optical_transformer_update_qstats
from torch import Tensor
from transformers.models.llama.modeling_llama import (
    LlamaAttention as HFLlamaAttention,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as HFLlamaDecoderLayer,
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)

logger = logging.getLogger(__name__)


def optical_transformer_SDPA(
    query,
    key,
    value,
    query_min_max: Tensor,
    key_min_max: Tensor,
    qk_min_max: Tensor,
    attn_min_max: Tensor,
    value_min_max: Tensor,
    av_min_max: Tensor,
    q_min_max_quantiles: Tensor,
    q_seed: Tensor,
    q_update_stats: bool,
    q_levels: int = 256,
    q_lut_min: float = 0.020040,
    q_smooth_factor: float = 0.9,
    q_bypass: bool = False,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    """
    Optical Transformer Scaled Dot-Product Attention.

    Computes scaled dot-product attention with quantized matrix multiplications
    to simulate optical neural network hardware constraints. This function applies
    quantization to both the query-key and attention-value matrix products.

    The quantization statistics (min/max values) are updated in-place during training
    using an exponential moving average controlled by ``q_smooth_factor``.

    Args:
        query (Tensor): Query tensor of shape ``(batch, heads, seq_len, head_dim)``.
        key (Tensor): Key tensor of shape ``(batch, kv_heads, seq_len, head_dim)``.
        value (Tensor): Value tensor of shape ``(batch, kv_heads, seq_len, head_dim)``.
        query_min_max (Tensor): Running min/max statistics for query quantization.
        key_min_max (Tensor): Running min/max statistics for key quantization.
        qk_min_max (Tensor): Running min/max statistics for query-key product.
        attn_min_max (Tensor): Min/max range for attention weights (typically [0, 1]).
        value_min_max (Tensor): Running min/max statistics for value quantization.
        av_min_max (Tensor): Running min/max statistics for attention-value product.
        q_min_max_quantiles (Tensor): Quantile values for computing statistics.
        q_seed (Tensor): Random seed for quantization noise.
        q_update_stats (bool): Whether to update running statistics.
        q_levels (int): Number of quantization levels. Default: 256.
        q_lut_min (float): Minimum value for lookup table. Default: 0.020040.
        q_smooth_factor (float): EMA smoothing factor for statistics. Default: 0.9.
        q_bypass (bool): If True, skip quantization. Default: False.
        attn_mask (Tensor, optional): Attention mask. Default: None.
        dropout_p (float): Dropout probability. Default: 0.0.
        is_causal (bool): If True, apply causal masking. Default: False.
        scale (float, optional): Scaling factor. If None, uses ``1/sqrt(head_dim)``.
        enable_gqa (bool): Enable grouped query attention. Default: False.

    Returns:
        Tensor: Attention output of shape ``(batch, heads, seq_len, head_dim)``.
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        # When is_causal=True, we construct a causal mask and ignore attn_mask
        # (similar to PyTorch's scaled_dot_product_attention behavior)
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    # attn_weight = query @ key.transpose(-2, -1) * scale_factor
    #
    if q_update_stats:
        with torch.no_grad():
            query_min_max_ = optical_transformer_update_qstats(
                query, query_min_max, q_min_max_quantiles, q_smooth_factor
            )
            query_min_max.copy_(query_min_max_)
            key_min_max_ = optical_transformer_update_qstats(
                key, key_min_max, q_min_max_quantiles, q_smooth_factor
            )
            key_min_max.copy_(key_min_max_)
            value_min_max_ = optical_transformer_update_qstats(
                value, value_min_max, q_min_max_quantiles, q_smooth_factor
            )
            value_min_max.copy_(value_min_max_)
            if not qk_min_max.isfinite().all():
                attn_weight_ = query @ key.transpose(-2, -1)
                attn_score_ = torch.softmax(
                    attn_weight_ * scale_factor + attn_bias, dim=-1
                )
                y_ = attn_score_ @ value
                qk_min_max_ = optical_transformer_update_qstats(
                    attn_weight_, qk_min_max, q_min_max_quantiles, q_smooth_factor
                )
                qk_min_max.copy_(qk_min_max_)
                # attn_min_max_ = _optical_transformer_update_stats(
                #     attn_score_, attn_min_max, query_min_max, q_smooth_factor
                # )
                # attn_min_max.copy_(attn_min_max_)
                av_min_max_ = optical_transformer_update_qstats(
                    y_, av_min_max, q_min_max_quantiles, q_smooth_factor
                )
                av_min_max.copy_(av_min_max_)

    attn_weight, _ = OTFunctions.quantized_matmul_fn(
        a=query.contiguous(),
        b=key.transpose(-2, -1).contiguous(),
        a_min=query_min_max[0].item(),
        a_max=query_min_max[1].item(),
        b_min=key_min_max[0].item(),
        b_max=key_min_max[1].item(),
        b_lut_min=q_lut_min,
        o_min=qk_min_max[0].item(),
        o_max=qk_min_max[1].item(),
        q_levels=q_levels,
        q_seed=q_seed.item(),
        skip_quantize=q_bypass,
    )
    if q_update_stats:
        with torch.no_grad():
            qk_min_max_ = optical_transformer_update_qstats(
                attn_weight, qk_min_max, q_min_max_quantiles, q_smooth_factor
            )
            qk_min_max.copy_(qk_min_max_)

    attn_weight = attn_weight * scale_factor
    #
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # y = attn_weight @ value
    #
    y, _ = OTFunctions.quantized_matmul_fn(
        a=attn_weight,
        b=value.contiguous(),
        a_min=attn_min_max[0].item(),
        a_max=attn_min_max[1].item(),
        b_min=value_min_max[0].item(),
        b_max=value_min_max[1].item(),
        b_lut_min=q_lut_min,
        o_min=av_min_max[0].item(),
        o_max=av_min_max[1].item(),
        q_levels=q_levels,
        q_seed=q_seed.item() + 1,
        skip_quantize=q_bypass,
    )
    if q_update_stats:
        with torch.no_grad():
            av_min_max_ = optical_transformer_update_qstats(
                y, av_min_max, q_min_max_quantiles, q_smooth_factor
            )
            av_min_max.copy_(av_min_max_)
    #
    return y


FakeModelArgs = namedtuple("FakeModelArgs", ["n_heads", "n_kv_heads", "dim"])


class OtLlamaAttention(nn.Module):
    """
    Optical Transformer attention module for LLaMA models.

    This module replaces the standard HuggingFace LlamaAttention with an optical
    transformer equivalent that simulates quantized matrix multiplications as would
    occur in optical neural network hardware. The implementation is based on the
    `Optical Transformers paper <https://arxiv.org/abs/2302.10360>`_.

    The attention computation uses optical transformer scaled dot-product attention
    (SDPA) which applies quantization to the query-key and attention-value matrix
    multiplications to simulate optical compute constraints.

    Args:
        config: HuggingFace LLaMA configuration object.
        layer_idx (int): Index of this attention layer in the model.
        q_levels (int): Number of quantization levels for optical simulation. Default: 256.
        q_lut_min (float): Minimum value for the lookup table used in quantization. Default: 0.020040.
        q_quantiles (tuple[float, float], optional): Quantile range for min/max statistics.
            If None, uses absolute min/max. Default: None.
        q_smooth_factor (float): Exponential moving average factor for updating
            running min/max statistics during training. Default: 0.9.
        q_init_seed (int): Random seed for quantization noise initialization. Default: 0.
        q_bypass (bool): If True, bypasses optical quantization and uses standard
            PyTorch attention. Useful for debugging or comparison. Default: False.

    Attributes:
        query_min_max (Tensor): Running min/max statistics for query tensors.
        key_min_max (Tensor): Running min/max statistics for key tensors.
        value_min_max (Tensor): Running min/max statistics for value tensors.
        qk_min_max (Tensor): Running min/max statistics for query-key products.
        attn_min_max (Tensor): Min/max range for attention weights (fixed at [0, 1]).
        av_min_max (Tensor): Running min/max statistics for attention-value products.
        seed (Tensor): Current random seed state for quantization.

    Example:
        .. code-block:: python

            from chop.passes.module.transforms.onn.layers.attn import OtLlamaAttention

            # Create from existing HuggingFace attention layer
            ot_attn = OtLlamaAttention.from_pretrained(
                hf_attention_layer,
                q_levels=256,
                q_bypass=False,
            )
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        q_levels: int = 256,
        q_lut_min: float = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_levels = q_levels
        self.q_lut_min = q_lut_min
        if q_quantiles is None:
            self.q_min_max_quantiles = None
        else:
            self.register_buffer("q_min_max_quantiles", torch.tensor(q_quantiles))
        self.register_buffer(
            "query_min_max", torch.tensor([float("inf"), float("-inf")])
        )
        self.register_buffer("key_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("qk_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("attn_min_max", torch.tensor([float(0), float(1)]))
        self.register_buffer(
            "value_min_max", torch.tensor([float("inf"), float("-inf")])
        )
        self.register_buffer("av_min_max", torch.tensor([float("inf"), float("-inf")]))
        self.register_buffer("seed", torch.tensor(q_init_seed, dtype=torch.int64))
        self.stat_smooth_factor = q_smooth_factor
        self.bypass = q_bypass

        self.query_min_max: Tensor
        self.key_min_max: Tensor
        self.qk_min_max: Tensor
        self.attn_min_max: Tensor
        self.value_min_max: Tensor
        self.av_min_max: Tensor
        self.seed: Tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = hf_apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = None
        if self.bypass:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=True,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                scale=self.scaling,
                attn_mask=attention_mask,
            )
        else:
            # Check if stats are uninitialized (inf)
            is_uninitialized = (
                self.query_min_max.isinf().any()
                or self.key_min_max.isinf().any()
                or self.value_min_max.isinf().any()
                or self.qk_min_max.isinf().any()
                or self.av_min_max.isinf().any()
            )

            attn_output = optical_transformer_SDPA(
                query_states,
                key_states,
                value_states,
                query_min_max=self.query_min_max,
                key_min_max=self.key_min_max,
                qk_min_max=self.qk_min_max,
                attn_min_max=self.attn_min_max,
                value_min_max=self.value_min_max,
                av_min_max=self.av_min_max,
                q_min_max_quantiles=self.q_min_max_quantiles,
                q_seed=self.seed,
                q_update_stats=self.training or is_uninitialized,
                q_levels=self.q_levels,
                q_lut_min=self.q_lut_min,
                q_smooth_factor=0.0 if is_uninitialized else self.stat_smooth_factor,
                q_bypass=self.bypass,
                is_causal=True,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                scale=self.scaling,
                attn_mask=attention_mask,
                enable_gqa=self.num_key_value_groups > 1,
            )
            with torch.no_grad():
                self.seed = self.seed + 2

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_pretrained(
        cls,
        attn: HFLlamaAttention,
        q_levels: int = 256,
        q_lut_min: float = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ):
        """
        Create an OtLlamaAttention module from a pretrained HuggingFace LlamaAttention.

        This factory method converts an existing LlamaAttention layer to its optical
        transformer equivalent, copying over the pretrained weights.

        Args:
            attn (HFLlamaAttention): The source HuggingFace LlamaAttention module.
            q_levels (int): Number of quantization levels. Default: 256.
            q_lut_min (float): Minimum value for lookup table. Default: 0.020040.
            q_quantiles (tuple[float, float], optional): Quantile range for statistics. Default: None.
            q_smooth_factor (float): Smoothing factor for running statistics. Default: 0.9.
            q_init_seed (int): Random seed for initialization. Default: 0.
            q_bypass (bool): If True, bypass optical quantization. Default: False.

        Returns:
            OtLlamaAttention: A new optical transformer attention module with copied weights.
        """
        new_attn = cls(
            attn.config,
            attn.layer_idx,
            q_levels=q_levels,
            q_lut_min=q_lut_min,
            q_quantiles=q_quantiles,
            q_smooth_factor=q_smooth_factor,
            q_init_seed=q_init_seed,
            q_bypass=q_bypass,
        )

        with torch.no_grad():
            if attn.q_proj.weight.device != torch.device("meta"):
                new_attn.load_state_dict(attn.state_dict(), strict=False)

        return new_attn
