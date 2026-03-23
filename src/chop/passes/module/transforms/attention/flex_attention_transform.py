"""
FlexAttention transform pass for MASE.

Walks a model and replaces standard attention modules with FlexAttention
variants that use ``torch.compile(flex_attention)`` with configurable
``score_mod`` and ``block_mask`` functions.

Supported model families:
    - Llama  (LlamaAttention / LlamaSdpaAttention)
    - Mistral (MistralAttention / MistralSdpaAttention)
    - BERT   (BertSelfAttention)

Usage::

    from chop.passes.module.transforms.attention import flex_attention_transform_pass

    pass_args = {
        "score_mod": "causal",                      # or "sliding_window", "alibi", "none"
        "score_mod_kwargs": {"window_size": 256},    # for parameterised score_mods
    }
    network, stats = flex_attention_transform_pass(network, pass_args)
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from .score_mods import get_score_mod, get_mask_mod
from ...module_modify_helper import set_module_by_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled flex_attention kernel (function-level compile, NOT whole-model)
# Deferred so that importing on CPU doesn't trigger compilation errors.
# ---------------------------------------------------------------------------
_compiled_flex_attention = None


# def _get_compiled_flex_attention():
#     """Return the compiled flex_attention kernel, compiling on first use.

#     Uses ``dynamic=False`` to avoid symbolic-shape issues in the inductor's
#     flex_decoding kernel (PyTorch 2.6 bug with ``get_split_k``).
#     """
#     global _compiled_flex_attention
#     if _compiled_flex_attention is None:
#         _compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
#     return _compiled_flex_attention


def _get_compiled_flex_attention():
    """Return the compiled flex_attention kernel, compiling on first use."""
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(
            flex_attention, 
            dynamic=False,
            mode="max-autotune-no-cudagraphs"
        )
    return _compiled_flex_attention


# FlexAttention's default block size for block-sparse tiling.
# Sequences shorter than this produce sub-block tiles that trigger
# Triton kernel assertion failures.  Block-level sparsity is also
# pointless when there is only one block, so we simply skip it.
_FLEX_BLOCK_SIZE = 128


def _make_block_mask(mask_mod_fn, Q_LEN, KV_LEN, device="cuda"):
    """
    Create a block_mask using the given mask_mod function.

    The block_mask tells FlexAttention which blocks to skip entirely,
    providing the main source of speedup for sparse attention patterns.

    Uses ``B=None, H=None`` because our mask_mod functions (causal,
    sliding_window) are batch- and head-independent.  This avoids
    redundant per-batch/per-head computation in ``create_block_mask``.

    Returns ``None`` (no block_mask) when either sequence dimension is
    shorter than the FlexAttention block size, since block-level
    sparsity cannot help with a single block.
    """
    if Q_LEN < _FLEX_BLOCK_SIZE or KV_LEN < _FLEX_BLOCK_SIZE:
        return None
    return create_block_mask(
        mask_mod_fn, B=None, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device,
    )


class _BlockMaskCache:
    """
    Caches a ``BlockMask`` and reuses it while (Q_LEN, KV_LEN) stay
    the same.  ``create_block_mask`` is expensive and the PyTorch docs
    explicitly recommend creating it once and reusing it.  During training
    with fixed-length batches the dimensions never change, so the mask is
    built exactly once.
    """

    def __init__(self):
        self._key = None
        self._mask = None

    def get(self, mask_mod_fn, Q_LEN, KV_LEN, device="cuda"):
        key = (Q_LEN, KV_LEN, str(device))
        if key == self._key and self._mask is not None:
            return self._mask
        mask = _make_block_mask(mask_mod_fn, Q_LEN, KV_LEN, device)
        self._key = key
        self._mask = mask
        return mask


# ============================================================================
# Llama FlexAttention
# ============================================================================
def _build_llama_flex_attention_cls():
    """
    Lazily import Llama classes and build the FlexAttention subclass.

    Returns the class, or *None* if Llama model files are not available.
    """
    try:
        from chop.models.llama.modeling_llama import (
            LlamaAttention,
            LlamaSdpaAttention,
            apply_rotary_pos_emb,
            repeat_kv,
        )
        from transformers.cache_utils import Cache
    except ImportError:
        return None, None, set()

    # Also match HuggingFace's own Llama classes (different Python objects
    # even though the code is similar).  In transformers >= 4.51 the SDPA
    # variant was folded into LlamaAttention, so we import what exists.
    target_classes = {LlamaAttention, LlamaSdpaAttention}
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaAttention as HfLlamaAttention,
        )
        target_classes.add(HfLlamaAttention)
    except ImportError:
        pass
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaSdpaAttention as HfLlamaSdpaAttention,
        )
        target_classes.add(HfLlamaSdpaAttention)
    except ImportError:
        pass

    class LlamaFlexAttention(LlamaAttention):
        """
        Llama attention using ``torch.compile(flex_attention)`` with a
        pluggable ``score_mod`` and optional ``block_mask``.
        Inherits all projections and RoPE from
        :class:`LlamaAttention` -- only the forward kernel dispatch changes.
        """

        def __init__(self, config, layer_idx: Optional[int] = None):
            super().__init__(config, layer_idx)
            self.score_mod_fn = None  # set by the transform pass
            self.mask_mod_fn = None   # set by the transform pass (for block_mask)
            self._block_mask_cache = _BlockMaskCache()

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ) -> torch.Tensor:
            # Fall back to eager if attention weights are requested
            if output_attentions:
                logger.warning_once(
                    "LlamaFlexAttention does not support output_attentions=True. "
                    "Falling back to eager attention."
                )
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            # ---- cache_position / position_ids defaults (from SdpaAttention) ----
            if cache_position is None:
                past_seen_tokens = (
                    past_key_value.get_seq_length() if past_key_value is not None else 0
                )
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + hidden_states.shape[1],
                    device=hidden_states.device,
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            bsz, q_len, _ = hidden_states.size()

            # ---- Q / K / V projections ----
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            # ---- RoPE ----
            # Use pre-computed position_embeddings from newer HF transformers
            # when available; otherwise compute internally (MASE models).
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            # ---- KV cache ----
            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            # ---- Contiguity (required by some backends) ----
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

            # ---- Build block_mask if mask_mod is available (cached) ----
            block_mask = None
            if self.mask_mod_fn is not None:
                block_mask = self._block_mask_cache.get(
                    self.mask_mod_fn,
                    Q_LEN=query_states.shape[2],
                    KV_LEN=key_states.shape[2],
                    device=query_states.device,
                )

            # ---- FlexAttention kernel ----
            # Use enable_gqa=True so FlexAttention handles GQA natively
            # instead of manually expanding KV heads with repeat_kv.
            flex_kwargs = {"score_mod": self.score_mod_fn}
            if block_mask is not None:
                flex_kwargs["block_mask"] = block_mask
            if self.num_key_value_groups > 1:
                flex_kwargs["enable_gqa"] = True

            attn_output = _get_compiled_flex_attention()(
                query_states,
                key_states,
                value_states,
                **flex_kwargs,
            )

            # ---- reshape & output projection ----
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)

            return attn_output, None

    return LlamaFlexAttention, LlamaAttention, target_classes


# ============================================================================
# Mistral FlexAttention
# ============================================================================
def _build_mistral_flex_attention_cls():
    """Lazily import Mistral classes and build the FlexAttention subclass."""
    try:
        from chop.models.mistral.modeling_mistral import (
            MistralAttention,
            MistralSdpaAttention,
            apply_rotary_pos_emb,
            repeat_kv,
        )
        from transformers.cache_utils import Cache
    except ImportError:
        return None, None, set()

    # Also match HuggingFace's own Mistral classes.
    target_classes = {MistralAttention, MistralSdpaAttention}
    try:
        from transformers.models.mistral.modeling_mistral import (
            MistralAttention as HfMistralAttention,
        )
        target_classes.add(HfMistralAttention)
    except ImportError:
        pass
    try:
        from transformers.models.mistral.modeling_mistral import (
            MistralSdpaAttention as HfMistralSdpaAttention,
        )
        target_classes.add(HfMistralSdpaAttention)
    except ImportError:
        pass

    class MistralFlexAttention(MistralAttention):
        """
        Mistral attention using ``torch.compile(flex_attention)`` with a
        pluggable ``score_mod`` and optional ``block_mask``.
        """

        def __init__(self, config, layer_idx: Optional[int] = None):
            super().__init__(config, layer_idx)
            self.score_mod_fn = None
            self.mask_mod_fn = None
            self._block_mask_cache = _BlockMaskCache()

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            if output_attentions:
                logger.warning_once(
                    "MistralFlexAttention does not support output_attentions=True. "
                    "Falling back to eager attention."
                )
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            bsz, q_len, _ = hidden_states.size()

            # ---- Q / K / V projections ----
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            # ---- RoPE ----
            # Use pre-computed position_embeddings from newer HF transformers
            # when available; otherwise compute internally (MASE models).
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            # ---- KV cache ----
            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            # ---- Contiguity ----
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

            # ---- Build block_mask if mask_mod is available (cached) ----
            block_mask = None
            if self.mask_mod_fn is not None:
                block_mask = self._block_mask_cache.get(
                    self.mask_mod_fn,
                    Q_LEN=query_states.shape[2],
                    KV_LEN=key_states.shape[2],
                    device=query_states.device,
                )

            # ---- FlexAttention kernel ----
            flex_kwargs = {"score_mod": self.score_mod_fn}
            if block_mask is not None:
                flex_kwargs["block_mask"] = block_mask
            if self.num_key_value_groups > 1:
                flex_kwargs["enable_gqa"] = True

            attn_output = _get_compiled_flex_attention()(
                query_states,
                key_states,
                value_states,
                **flex_kwargs,
            )

            # ---- reshape & output projection ----
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)

            return attn_output, None

    return MistralFlexAttention, MistralAttention, target_classes


# ============================================================================
# BERT FlexAttention
# ============================================================================
def _build_bert_flex_attention_cls():
    """Lazily import BERT classes and build the FlexAttention subclass."""
    try:
        from chop.models.bert.modeling_bert import BertSelfAttention
    except ImportError:
        return None, None, set()

    class BertFlexSelfAttention(BertSelfAttention):
        """
        BERT self-attention using ``torch.compile(flex_attention)`` with a
        pluggable ``score_mod`` and optional ``block_mask``.
        """

        def __init__(self, config, position_embedding_type=None):
            super().__init__(config, position_embedding_type)
            self.score_mod_fn = None
            self.mask_mod_fn = None
            self._block_mask_cache = _BlockMaskCache()

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
        ) -> torch.Tensor:
            # Cross-attention and relative position embeddings are not
            # supported by the flex path -- fall back to eager.
            if (
                encoder_hidden_states is not None
                or self.position_embedding_type != "absolute"
                or output_attentions
            ):
                return super().forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )

            # ---- Q / K / V projections ----
            mixed_query_layer = self.query(hidden_states)

            if past_key_value is not None:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))

            query_layer = self.transpose_for_scores(mixed_query_layer)

            # ---- Contiguity ----
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

            # ---- Build block_mask if mask_mod is available (cached) ----
            block_mask = None
            if self.mask_mod_fn is not None:
                block_mask = self._block_mask_cache.get(
                    self.mask_mod_fn,
                    Q_LEN=query_layer.shape[2],
                    KV_LEN=key_layer.shape[2],
                    device=query_layer.device,
                )

            # ---- FlexAttention kernel ----
            flex_kwargs = {"score_mod": self.score_mod_fn}
            if block_mask is not None:
                flex_kwargs["block_mask"] = block_mask

            context_layer = _get_compiled_flex_attention()(
                query_layer,
                key_layer,
                value_layer,
                **flex_kwargs,
            )

            # ---- reshape ----
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (
                self.all_head_size,
            )
            context_layer = context_layer.view(new_context_layer_shape)

            return context_layer

    target_classes = {BertSelfAttention}
    return BertFlexSelfAttention, BertSelfAttention, target_classes


# ============================================================================
# Module replacement logic
# ============================================================================

def _replace_attention_modules(network, flex_cls, base_cls, target_classes, score_mod_fn, mask_mod_fn):
    """
    Walk *network*, find every module whose type is in *target_classes*,
    and replace it with an instance of *flex_cls* carrying the same weights.

    Returns:
        (network, count) -- the mutated network and how many modules were replaced.
    """
    replacements = []

    for name, module in network.named_modules():
        if type(module) in target_classes:
            replacements.append((name, module))

    for name, old_module in replacements:
        # Instantiate the flex variant with the same constructor args
        if hasattr(old_module, "config"):
            # Llama / Mistral style
            new_module = flex_cls(
                config=old_module.config,
                layer_idx=getattr(old_module, "layer_idx", None),
            )
        else:
            # BERT style -- reconstruct config from stored attributes.
            bert_cfg_attrs = {
                "num_attention_heads": old_module.num_attention_heads,
                "hidden_size": old_module.all_head_size,
                "attention_probs_dropout_prob": old_module.dropout.p,
                "is_decoder": getattr(old_module, "is_decoder", False),
            }
            if hasattr(old_module, "max_position_embeddings"):
                bert_cfg_attrs["max_position_embeddings"] = (
                    old_module.max_position_embeddings
                )
            bert_config = type("BertConfigProxy", (), bert_cfg_attrs)()

            new_module = flex_cls(
                config=bert_config,
                position_embedding_type=getattr(
                    old_module, "position_embedding_type", "absolute"
                ),
            )

        # Transfer weights (same architecture, so state_dict keys match)
        new_module.load_state_dict(old_module.state_dict())
        new_module.score_mod_fn = score_mod_fn
        new_module.mask_mod_fn = mask_mod_fn

        # Move to same device / dtype
        device = next(old_module.parameters()).device
        dtype = next(old_module.parameters()).dtype
        new_module = new_module.to(device=device, dtype=dtype)

        network = set_module_by_name(network, name, new_module)

    return network, len(replacements)


# ============================================================================
# Public pass function
# ============================================================================

def flex_attention_transform_pass(network, pass_args):
    """
    Replace standard attention modules with FlexAttention variants.

    :param network: The model (``nn.Module``) to transform.
    :param pass_args: Configuration dictionary.

    ``pass_args`` keys:

    * ``score_mod`` (str, default ``"causal"``): name of the score
      modification function -- one of ``"none"``, ``"causal"``,
      ``"sliding_window"``, ``"alibi"``.
    * ``score_mod_kwargs`` (dict, default ``{}``): extra keyword
      arguments forwarded to parameterised score_mod factories
      (e.g. ``{"window_size": 256}``).
    * ``use_block_mask`` (bool, default ``True``): whether to create
      a matching block_mask for block-level sparsity. Set to ``False``
      to disable (e.g. for benchmarking without block skipping).

    The pass automatically creates a matching ``mask_mod`` (block_mask)
    for score_mods that have a corresponding mask pattern ("causal",
    "sliding_window"). This enables block-level sparsity for speedup.

    :returns: ``(network, stats)`` where *stats* is a dict with counts
        of replaced modules per model family.

    Example::

        pass_args = {
            "score_mod": "sliding_window",
            "score_mod_kwargs": {"window_size": 256},
        }
        model, stats = flex_attention_transform_pass(model, pass_args)
    """
    score_mod_name = pass_args.get("score_mod", "causal")
    score_mod_kwargs = pass_args.get("score_mod_kwargs", {})
    use_block_mask = pass_args.get("use_block_mask", True)

    score_mod_fn = get_score_mod(score_mod_name, **score_mod_kwargs)

    # Automatically create matching mask_mod for block-level sparsity
    mask_mod_fn = None
    if use_block_mask:
        try:
            mask_mod_fn = get_mask_mod(score_mod_name, **score_mod_kwargs)
        except ValueError:
            # No matching mask_mod for this score_mod (e.g. alibi)
            # That's fine -- just use score_mod without block_mask
            logger.info(
                f"No mask_mod available for '{score_mod_name}', "
                f"running without block_mask (no block-level sparsity)."
            )

    stats = {}

    # --- Llama ---
    llama_flex_cls, llama_base_cls, llama_targets = _build_llama_flex_attention_cls()
    if llama_flex_cls is not None:
        network, count = _replace_attention_modules(
            network, llama_flex_cls, llama_base_cls, llama_targets, score_mod_fn, mask_mod_fn
        )
        stats["llama_replaced"] = count

    # --- Mistral ---
    mistral_flex_cls, mistral_base_cls, mistral_targets = _build_mistral_flex_attention_cls()
    if mistral_flex_cls is not None:
        network, count = _replace_attention_modules(
            network, mistral_flex_cls, mistral_base_cls, mistral_targets, score_mod_fn, mask_mod_fn
        )
        stats["mistral_replaced"] = count

    # --- BERT ---
    bert_flex_cls, bert_base_cls, bert_targets = _build_bert_flex_attention_cls()
    if bert_flex_cls is not None:
        network, count = _replace_attention_modules(
            network, bert_flex_cls, bert_base_cls, bert_targets, score_mod_fn, mask_mod_fn
        )
        stats["bert_replaced"] = count

    total = sum(stats.values())
    block_mask_status = "with block_mask" if mask_mod_fn is not None else "without block_mask"
    logger.info(
        f"flex_attention_transform_pass: replaced {total} attention module(s) "
        f"with score_mod='{score_mod_name}' ({block_mask_status}). Details: {stats}"
    )

    return network, stats
