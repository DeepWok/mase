from typing import Optional

import torch
from mase_triton.optical_compute import OpticalTransformerFunctions as OTFunctions
from mase_triton.optical_compute.layers import OpticalTransformerLinear as OTLinear
from mase_triton.optical_compute.layers import optical_transformer_update_qstats
from mase_triton.utils.torch_module import get_layer_name, set_layer_by_name
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)


def ot_eager_attention_forward(
    module: "OtLlamaAttention",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
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
        attention_mask (Tensor, optional): Attention mask. Default: None.
        dropout (float): Dropout probability. Default: 0.0.
        scaling (float, optional): Scaling factor. If None, uses ``1/sqrt(head_dim)``.

    Returns:
        Tensor: Attention output of shape ``(batch, heads, seq_len, head_dim)``.
    """
    with torch.no_grad():
        query_min_max_ = optical_transformer_update_qstats(
            query,
            module.query_min_max,
            module.q_min_max_quantiles,
            module.stat_smooth_factor,
        )
        module.query_min_max.copy_(query_min_max_)
        key_min_max_ = optical_transformer_update_qstats(
            key,
            module.key_min_max,
            module.q_min_max_quantiles,
            module.stat_smooth_factor,
        )
        module.key_min_max.copy_(key_min_max_)
        key_states = repeat_kv(key, module.num_key_value_groups)
        if not module.qk_min_max.isfinite().all():
            attn_weights = torch.matmul(query, key_states.transpose(-1, -2)) * scaling
            qk_min_max_ = optical_transformer_update_qstats(
                attn_weights,
                module.qk_min_max,
                module.q_min_max_quantiles,
                module.stat_smooth_factor,
            )
            module.qk_min_max.copy_(qk_min_max_)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights, _ = OTFunctions.quantized_matmul_fn(
        a=query.contiguous(),
        b=key_states.transpose(2, 3).contiguous(),
        a_min=module.query_min_max[0],
        a_max=module.query_min_max[1],
        b_min=module.key_min_max[0],
        b_max=module.key_min_max[1],
        b_lut_min=module.q_lut_min,
        o_min=module.qk_min_max[0],
        o_max=module.qk_min_max[1],
        q_levels=module.q_levels,
        q_seed=module.seed.item(),
        skip_quantize=False,
    )
    attn_weights = attn_weights * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    # attn_output = torch.matmul(attn_weights, value_states)

    with torch.no_grad():
        attn_min_max_ = optical_transformer_update_qstats(
            attn_weights,
            module.attn_min_max,
            module.q_min_max_quantiles,
            module.stat_smooth_factor,
        )
        module.attn_min_max.copy_(attn_min_max_)
        value_min_max_ = optical_transformer_update_qstats(
            value_states,
            module.value_min_max,
            module.q_min_max_quantiles,
            module.stat_smooth_factor,
        )
        module.value_min_max.copy_(value_min_max_)
        attn_ = torch.matmul(attn_weights, value_states)
        av_min_max_ = optical_transformer_update_qstats(
            attn_,
            module.av_min_max,
            module.q_min_max_quantiles,
            module.stat_smooth_factor,
        )
        module.av_min_max.copy_(av_min_max_)

    attn_output, _ = OTFunctions.quantized_matmul_fn(
        a=attn_weights.contiguous(),
        b=value_states.contiguous(),
        a_min=module.attn_min_max[0],
        a_max=module.attn_min_max[1],
        b_min=module.value_min_max[0],
        b_max=module.value_min_max[1],
        b_lut_min=module.q_lut_min,
        o_min=module.av_min_max[0],
        o_max=module.av_min_max[1],
        q_levels=module.q_levels,
        q_seed=module.seed.item(),
        skip_quantize=module.bypass,
    )
    with torch.no_grad():
        module.seed += 1

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


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
                layer_idx=0,
                q_levels=256,
                q_bypass=False,
            )
    """

    def __init__(
        self,
        config: LlamaConfig,
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
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if self.bypass:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        else:
            attn_output, attn_weights = ot_eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            self.seed += 1

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_pretrained(
        cls,
        attn: LlamaAttention,
        layer_idx: int,
        q_levels: int = 256,
        q_lut_min: float = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ) -> "OtLlamaAttention":
        assert isinstance(attn, LlamaAttention)
        ot_attn = cls(
            attn.config,
            layer_idx,
            q_levels,
            q_lut_min,
            q_quantiles,
            q_smooth_factor,
            q_init_seed,
            q_bypass,
        )
        ot_attn.to(attn.o_proj.weight.dtype)
        ot_attn.load_state_dict(attn.state_dict(), strict=False)
        return ot_attn
