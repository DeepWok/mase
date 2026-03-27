"""
Quantized LLaDA block with MXFP/MXInt KV cache, RoPE, softmax, and matmul quantization.

LLaDA differs from Llama/Qwen in that attention is not a separate module —
it's a method inside LLaDALlamaBlock. So we quantize the block itself,
overriding the `attention` method to inject quantization on QK/AV matmul,
RoPE, softmax, and KV cache.
"""

from typing import Optional, Tuple
from functools import partial

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .modeling_llada import LLaDALlamaBlock, BufferCache, ModelConfig
from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.nn.quantized.functional.kvcache import kv_cache_mxfp, kv_cache_mxint


class LLaDABlockMXFP(LLaDALlamaBlock):
    """MXFP-quantized LLaDA block with KV cache, RoPE, softmax, and matmul quantization."""

    def __init__(
        self, layer_id, config: ModelConfig, cache: BufferCache, q_config: dict = None
    ):
        super().__init__(layer_id, config, cache)
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

    def attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, T, C = q.size()
        dtype = k.dtype

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(
            B, T, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(1, 2)
        v = v.view(
            B, T, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            if replace_position is None:
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)
            else:
                B_rp = replace_position.shape[0]
                for batch_idx in range(B_rp):
                    batch_replace_indices = replace_position[batch_idx].nonzero(
                        as_tuple=True
                    )[0]
                    if len(batch_replace_indices) > 0:
                        past_key[batch_idx, :, batch_replace_indices] = k[
                            batch_idx, :, : len(batch_replace_indices)
                        ]
                        past_value[batch_idx, :, batch_replace_indices] = v[
                            batch_idx, :, : len(batch_replace_indices)
                        ]
                k = past_key
                v = past_value

        # KV cache quantization
        if use_cache and not self.kv_cache_bypass:
            k, v = kv_cache_mxfp(k, v, config=self.kv_cache_config)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]

        if self.config.rope:
            if replace_position is None:
                q, k = self.rotary_emb(q, k)
            else:
                max_replace_pos = (
                    replace_position.nonzero(as_tuple=True)[1].max() + 1
                    if replace_position.any()
                    else key_len
                )
                q, k = self.rotary_emb(q, k, max_replace_pos)

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # QK matmul quantization
        assert k.size(1) == v.size(1)
        num_kv_heads = k.size(1)
        num_q_heads = q.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            k_expanded = k.repeat_interleave(
                num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads
            )
            v_expanded = v.repeat_interleave(
                num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads
            )
        else:
            k_expanded, v_expanded = k, v

        if not self.qk_bypass:
            q_quantizer = partial(
                mxfp_quantizer,
                block_size=self.qk_config["data_in_block_size"],
                element_exp_bits=self.qk_config["data_in_exponent_width"],
                element_frac_bits=self.qk_config["data_in_frac_width"],
                block_dim=-1,
            )
            q = q_quantizer(q)

        head_dim = C // self.config.n_heads
        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) / (head_dim**0.5)

        if attention_bias is not None:
            attn_weights = attn_weights + attention_bias

        # Softmax quantization
        if not self.softmax_bypass:
            from chop.nn.quantized.functional.softmax import softmax_minifloat

            attn_weights = softmax_minifloat(attn_weights, self.softmax_config, dim=-1)
        else:
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)

        attn_weights = F.dropout(
            attn_weights, p=0.0 if not self.training else self.config.attention_dropout
        )

        # AV matmul quantization
        if not self.av_bypass:
            a_quantizer = partial(
                mxfp_quantizer,
                block_size=self.av_config["data_in_block_size"],
                element_exp_bits=self.av_config["data_in_exponent_width"],
                element_frac_bits=self.av_config["data_in_frac_width"],
                block_dim=-1,
            )
            attn_weights = a_quantizer(attn_weights)

        att = torch.matmul(attn_weights, v_expanded)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_out(att), present


class LLaDABlockMXInt(LLaDALlamaBlock):
    """MXInt-quantized LLaDA block with KV cache, RoPE, softmax, and matmul quantization."""

    def __init__(
        self, layer_id, config: ModelConfig, cache: BufferCache, q_config: dict = None
    ):
        super().__init__(layer_id, config, cache)
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

    def attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, T, C = q.size()
        dtype = k.dtype

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(
            B, T, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(1, 2)
        v = v.view(
            B, T, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            if replace_position is None:
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)
            else:
                B_rp = replace_position.shape[0]
                for batch_idx in range(B_rp):
                    batch_replace_indices = replace_position[batch_idx].nonzero(
                        as_tuple=True
                    )[0]
                    if len(batch_replace_indices) > 0:
                        past_key[batch_idx, :, batch_replace_indices] = k[
                            batch_idx, :, : len(batch_replace_indices)
                        ]
                        past_value[batch_idx, :, batch_replace_indices] = v[
                            batch_idx, :, : len(batch_replace_indices)
                        ]
                k = past_key
                v = past_value

        # KV cache quantization
        if use_cache and not self.kv_cache_bypass:
            k, v = kv_cache_mxint(k, v, config=self.kv_cache_config)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]

        if self.config.rope:
            if replace_position is None:
                q, k = self.rotary_emb(q, k)
            else:
                max_replace_pos = (
                    replace_position.nonzero(as_tuple=True)[1].max() + 1
                    if replace_position.any()
                    else key_len
                )
                q, k = self.rotary_emb(q, k, max_replace_pos)

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # GQA expansion
        assert k.size(1) == v.size(1)
        num_kv_heads = k.size(1)
        num_q_heads = q.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            k_expanded = k.repeat_interleave(
                num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads
            )
            v_expanded = v.repeat_interleave(
                num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads
            )
        else:
            k_expanded, v_expanded = k, v

        # QK matmul quantization
        if not self.qk_bypass:
            q_quantizer = partial(
                mxint_quantizer,
                block_size=self.qk_config["data_in_block_size"],
                element_bits=self.qk_config["data_in_width"],
                block_dim=-1,
            )
            q = q_quantizer(q)

        head_dim = C // self.config.n_heads
        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) / (head_dim**0.5)

        if attention_bias is not None:
            attn_weights = attn_weights + attention_bias

        # Softmax quantization
        if not self.softmax_bypass:
            from chop.nn.quantized.functional.softmax import softmax_minifloat

            attn_weights = softmax_minifloat(attn_weights, self.softmax_config, dim=-1)
        else:
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)

        attn_weights = F.dropout(
            attn_weights, p=0.0 if not self.training else self.config.attention_dropout
        )

        # AV matmul quantization
        if not self.av_bypass:
            a_quantizer = partial(
                mxint_quantizer,
                block_size=self.av_config["data_in_block_size"],
                element_bits=self.av_config["data_in_width"],
                block_dim=-1,
            )
            attn_weights = a_quantizer(attn_weights)

        att = torch.matmul(attn_weights, v_expanded)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_out(att), present
