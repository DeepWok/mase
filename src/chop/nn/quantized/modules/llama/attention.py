from typing import Optional, Tuple

import torch
from torch import Tensor, nn, LongTensor
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaConfig,
    Cache,
    repeat_kv,
    LlamaAttention,
    eager_attention_forward as _hf_eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from functools import partial

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.nn.quantized.functional.rope import rope_minifloat
from chop.nn.quantized.functional.softmax import softmax_minifloat
from chop.nn.quantized.functional.kvcache import (
    kv_cache_mxfp,
    kv_cache_mxint,
    kv_cache_mxint_rotate,
    kv_cache_mxfp_rotate,
)
from chop.nn.quantized.functional.attention import (
    eager_attention_forward_mxfp as _eager_attention_forward_mxfp,
    eager_attention_forward_mxint as _eager_attention_forward_mxint,
    eager_attention_forward_mxint_rotate as _eager_attention_forward_mxint_rotate,
    eager_attention_forward_mxfp_rotate as _eager_attention_forward_mxfp_rotate,
)

import logging

logger = logging.getLogger(__name__)


def _hf_attention_dispatch(
    module,
    query_states,
    key_states,
    value_states,
    attention_mask,
    **kwargs,
):
    """Call HF's configured attention backend (sdpa / flash / eager / ...).

    Used when all in-attention quant stages (qk / av / softmax) are bypassed —
    in that case the wrapper has nothing to inject inside the attention compute,
    so we shouldn't force eager. Mirrors HF Llama's own dispatch line.
    """
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation, _hf_eager_attention_forward
    )
    return attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        **kwargs,
    )


class LlamaAttentionLSQInteger(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, q_config: dict = None):
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

        self.query_quan = LSQInteger(level=q_config["level"], sym=True)
        self.key_quan = LSQInteger(level=q_config["level"], sym=True)
        self.value_quan = LSQInteger(level=q_config["level"], sym=True)
        self.o_quant = LSQInteger(level=q_config["level"], sym=True)
        self.attn_quan = LSQInteger(level=q_config["level"], sym=False)
        self.after_attn_quan = LSQInteger(level=q_config["level"], sym=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.query_quan(query_states)
        key_states = self.key_quan(key_states)
        value_states = self.value_quan(value_states)

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

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )
        attn_weights = self.attn_quan(attn_weights)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self.after_attn_quan(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = self.o_quant(attn_output)

        return attn_output, attn_weights

    @classmethod
    def from_self(cls, attention: LlamaAttention, q_config: dict = None):
        new_attn = cls(
            config=attention.config,
            layer_idx=attention.layer_idx,
            q_config=q_config,
        )
        device, dtype = (
            next(attention.parameters()).device,
            next(attention.parameters()).dtype,
        )
        new_attn = new_attn.to(dtype=dtype, device=device)
        # strict=False: LSQInteger quantizer submodules add params not in base LlamaAttention
        new_attn.load_state_dict(attention.state_dict(), strict=False)
        return new_attn


class LlamaAttentionMXFP(LlamaAttention):
    """MXFP-quantized LlamaAttention."""

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx)
        q_config = q_config or {}
        self.qk_config = q_config.get("qk_matmul", {})
        self.av_config = q_config.get("av_matmul", {})
        self.rope_config = q_config.get("rope", {})
        self.softmax_config = q_config.get("softmax", {})
        self.kv_cache_config = q_config.get("kv_cache", {})
        self.qk_bypass = self.qk_config.get("bypass", False)
        self.av_bypass = self.av_config.get("bypass", False)
        self.rope_bypass = self.rope_config.get("bypass", False)
        self.softmax_bypass = self.softmax_config.get("bypass", False)
        self.kv_cache_bypass = self.kv_cache_config.get("bypass", False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                self.rope_config,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if not self.kv_cache_bypass:
                key_states, value_states = kv_cache_mxfp(
                    key_states,
                    value_states,
                    self.kv_cache_config,
                )
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        # Skip-replace: if no in-attention quant stage is active, fall back to
        # whatever attention backend HF was configured for (sdpa / fa2 / eager).
        # KV-cache / RoPE quant happens above and is unaffected.
        if self.qk_bypass and self.av_bypass and self.softmax_bypass:
            attn_output, attn_weights = _hf_attention_dispatch(
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
            assert self.config._attn_implementation == "eager", (
                "MXFP-quantized eager attention requires _attn_implementation='eager' "
                "when any of qk/av/softmax stages are active."
            )
            attn_output, attn_weights = _eager_attention_forward_mxfp(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                qk_bypass=self.qk_bypass,
                qk_config=self.qk_config,
                av_bypass=self.av_bypass,
                av_config=self.av_config,
                softmax_bypass=self.softmax_bypass,
                softmax_config=self.softmax_config,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_self(cls, attention: LlamaAttention, q_config: dict = None):
        new_attn = cls(
            config=attention.config,
            layer_idx=attention.layer_idx,
            q_config=q_config,
        )
        device, dtype = (
            next(attention.parameters()).device,
            next(attention.parameters()).dtype,
        )
        new_attn = new_attn.to(dtype=dtype, device=device)
        new_attn.load_state_dict(attention.state_dict(), strict=True)
        return new_attn


class LlamaAttentionMXInt(LlamaAttention):
    """MXInt-quantized LlamaAttention."""

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx)
        q_config = q_config or {}
        self.qk_config = q_config.get("qk_matmul", {})
        self.av_config = q_config.get("av_matmul", {})
        self.rope_config = q_config.get("rope", {})
        self.softmax_config = q_config.get("softmax", {})
        self.kv_cache_config = q_config.get("kv_cache", {})
        self.qk_bypass = self.qk_config.get("bypass", False)
        self.av_bypass = self.av_config.get("bypass", False)
        self.rope_bypass = self.rope_config.get("bypass", False)
        self.softmax_bypass = self.softmax_config.get("bypass", False)
        self.kv_cache_bypass = self.kv_cache_config.get("bypass", False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                self.rope_config,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if not self.kv_cache_bypass:
                key_states, value_states = kv_cache_mxint(
                    key_states,
                    value_states,
                    self.kv_cache_config,
                )
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        if self.qk_bypass and self.av_bypass and self.softmax_bypass:
            attn_output, attn_weights = _hf_attention_dispatch(
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
            assert self.config._attn_implementation == "eager", (
                "MXInt-quantized eager attention requires _attn_implementation='eager' "
                "when any of qk/av/softmax stages are active."
            )
            attn_output, attn_weights = _eager_attention_forward_mxint(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                qk_bypass=self.qk_bypass,
                qk_config=self.qk_config,
                av_bypass=self.av_bypass,
                av_config=self.av_config,
                softmax_bypass=self.softmax_bypass,
                softmax_config=self.softmax_config,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_self(cls, attention: LlamaAttention, q_config: dict = None):
        new_attn = cls(
            config=attention.config,
            layer_idx=attention.layer_idx,
            q_config=q_config,
        )
        device, dtype = (
            next(attention.parameters()).device,
            next(attention.parameters()).dtype,
        )
        new_attn = new_attn.to(dtype=dtype, device=device)
        new_attn.load_state_dict(attention.state_dict(), strict=True)
        return new_attn



class LlamaAttentionMXIntRotate(LlamaAttentionMXInt):
    """LlamaAttentionMXInt with online Hadamard rotation around the
    activation quantizers (KV-cache, Q@K, A@V). Mirrors
    ``Qwen3AttentionMXIntRotate``.

    Per-stage rotate toggles let the rotation search evaluate qk_matmul /
    av_matmul / kv_cache rotations independently within a single attention
    class. Each flag defaults to True so an unmodified rotate config keeps
    the original "all three rotated" behavior. Toggle via the matching block
    in q_config:

        q_config["qk_matmul"]["rotate"]   = True | False  (default True)
        q_config["av_matmul"]["rotate"]   = True | False  (default True)
        q_config["kv_cache"]["rotate"]    = True | False  (default True)
    """

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx, q_config=q_config)
        self.qk_use_rotate = self.qk_config.get("rotate", True)
        self.av_use_rotate = self.av_config.get("rotate", True)
        self.kv_cache_use_rotate = self.kv_cache_config.get("rotate", True)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                self.rope_config,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if not self.kv_cache_bypass:
                if self.kv_cache_use_rotate:
                    key_states, value_states = kv_cache_mxint_rotate(
                        key_states,
                        value_states,
                        self.kv_cache_config,
                    )
                else:
                    key_states, value_states = kv_cache_mxint(
                        key_states,
                        value_states,
                        self.kv_cache_config,
                    )
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        if self.qk_bypass and self.av_bypass and self.softmax_bypass:
            attn_output, attn_weights = _hf_attention_dispatch(
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
            assert self.config._attn_implementation == "eager", (
                "MXInt-rotate-quantized attention requires _attn_implementation='eager' "
                "when any of qk/av/softmax stages are active."
            )
            attn_output, attn_weights = _eager_attention_forward_mxint_rotate(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                qk_bypass=self.qk_bypass,
                qk_config=self.qk_config,
                av_bypass=self.av_bypass,
                av_config=self.av_config,
                softmax_bypass=self.softmax_bypass,
                softmax_config=self.softmax_config,
                qk_use_rotate=self.qk_use_rotate,
                av_use_rotate=self.av_use_rotate,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaAttentionMXFPRotate(LlamaAttentionMXFP):
    """LlamaAttentionMXFP with online Hadamard rotation around the activation
    quantizers (KV-cache, Q@K, A@V). MXFP analogue of ``LlamaAttentionMXIntRotate``.

    Per-stage rotate toggles (default True) let the rotation search evaluate
    qk_matmul / av_matmul / kv_cache rotations independently:

        q_config["qk_matmul"]["rotate"]   = True | False  (default True)
        q_config["av_matmul"]["rotate"]   = True | False  (default True)
        q_config["kv_cache"]["rotate"]    = True | False  (default True)
    """

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx, q_config=q_config)
        self.qk_use_rotate = self.qk_config.get("rotate", True)
        self.av_use_rotate = self.av_config.get("rotate", True)
        self.kv_cache_use_rotate = self.kv_cache_config.get("rotate", True)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                self.rope_config,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if not self.kv_cache_bypass:
                if self.kv_cache_use_rotate:
                    key_states, value_states = kv_cache_mxfp_rotate(
                        key_states,
                        value_states,
                        self.kv_cache_config,
                    )
                else:
                    key_states, value_states = kv_cache_mxfp(
                        key_states,
                        value_states,
                        self.kv_cache_config,
                    )
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        if self.qk_bypass and self.av_bypass and self.softmax_bypass:
            attn_output, attn_weights = _hf_attention_dispatch(
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
            assert self.config._attn_implementation == "eager", (
                "MXFP-rotate-quantized attention requires _attn_implementation='eager' "
                "when any of qk/av/softmax stages are active."
            )
            attn_output, attn_weights = _eager_attention_forward_mxfp_rotate(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                qk_bypass=self.qk_bypass,
                qk_config=self.qk_config,
                av_bypass=self.av_bypass,
                av_config=self.av_config,
                softmax_bypass=self.softmax_bypass,
                softmax_config=self.softmax_config,
                qk_use_rotate=self.qk_use_rotate,
                av_use_rotate=self.av_use_rotate,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
