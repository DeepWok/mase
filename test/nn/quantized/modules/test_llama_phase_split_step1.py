"""Step-1 phase-split regression tests for quantized Llama integration.

These tests intentionally stay small and deterministic:
- no model downloads
- no full-model forward
- only boundary behavior needed to protect step-1 invariants
"""

from __future__ import annotations

import sys
import types
import importlib.util
import inspect
from pathlib import Path

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Keep unit tests lightweight in environments without tensorboard dependency.
if "torch.utils.tensorboard" not in sys.modules:
    tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
    tensorboard_stub.SummaryWriter = object
    sys.modules["torch.utils.tensorboard"] = tensorboard_stub
if "cvxpy" not in sys.modules:
    sys.modules["cvxpy"] = types.ModuleType("cvxpy")

# Bypass heavyweight `chop/__init__.py` side effects during focused unit tests.
repo_root = next(p for p in Path(__file__).resolve().parents if (p / "src/chop").exists())
if "chop" not in sys.modules:
    chop_stub = types.ModuleType("chop")
    chop_stub.__path__ = [str(repo_root / "src/chop")]
    sys.modules["chop"] = chop_stub

_helper_spec = importlib.util.spec_from_file_location(
    "mase_module_modify_helper",
    repo_root / "src/chop/passes/module/module_modify_helper.py",
)
_helper_module = importlib.util.module_from_spec(_helper_spec)
assert _helper_spec is not None and _helper_spec.loader is not None
_helper_spec.loader.exec_module(_helper_module)
weight_replacement = _helper_module.weight_replacement

from chop.nn.quantized.modules.linear import LinearMXFP
from chop.nn.quantized.modules.llama.attention import LlamaAttentionMXFP
from chop.nn.quantized.modules.phase_config import normalize_phase_q_config
from chop.nn.quantized.modules.phase_context import (
    set_active_phase,
    infer_phase_from_hidden_and_cache,
    get_active_phase,
    infer_phase_from_decoder_layer_inputs,
)
from chop.passes.module.transforms.quantize.quantize import (
    _llama_decoder_layer_phase_pre_hook,
)


class _DummyCache:
    """Small cache stub for phase inference tests."""

    def __init__(self, seq_len: int):
        self._seq_len = seq_len

    def get_seq_length(self) -> int:
        return self._seq_len


def test_normalize_phase_q_config_legacy_compatibility():
    """Legacy flat config should map to both prefill/decode buckets."""

    legacy = {"data_in_block_size": 16, "bypass": False}
    normalized = normalize_phase_q_config(legacy)
    assert normalized["decode_policy"] == "fp_only"
    assert normalized["prefill"]["data_in_block_size"] == 16
    assert normalized["decode"]["data_in_block_size"] == 16


def test_infer_runtime_phase_from_cache_state():
    """Shared phase inference should follow cache semantics."""

    hidden_prefill = torch.randn(1, 8, 16)
    hidden_decode = torch.randn(1, 1, 16)

    assert infer_phase_from_hidden_and_cache(hidden_prefill, None) == "prefill"
    assert (
        infer_phase_from_hidden_and_cache(hidden_prefill, _DummyCache(seq_len=0))
        == "prefill"
    )
    assert (
        infer_phase_from_hidden_and_cache(hidden_decode, _DummyCache(seq_len=0))
        == "decode"
    )
    assert (
        infer_phase_from_hidden_and_cache(hidden_prefill, _DummyCache(seq_len=32))
        == "decode"
    )


def test_decoder_layer_pre_hook_sets_phase_before_input_layernorm():
    """Pre-hook should set phase before `input_layernorm` executes."""

    cfg = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    layer = LlamaDecoderLayer(cfg, layer_idx=0)

    observed = {"phase_at_input_ln": None}
    orig_ln_forward = layer.input_layernorm.forward

    def _spy_input_ln(hidden_states):
        observed["phase_at_input_ln"] = get_active_phase()
        return orig_ln_forward(hidden_states)

    layer.input_layernorm.forward = _spy_input_ln

    # Keep attention/MLP simple to avoid requiring full model plumbing.
    class _DummySelfAttn(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states, None

    class _DummyMLP(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    layer.self_attn = _DummySelfAttn()
    layer.mlp = _DummyMLP()
    layer.register_forward_pre_hook(_llama_decoder_layer_phase_pre_hook, with_kwargs=True)

    hidden = torch.randn(1, 1, cfg.hidden_size)
    set_active_phase("prefill")
    layer(
        hidden_states=hidden,
        past_key_values=_DummyCache(seq_len=8),
    )

    assert observed["phase_at_input_ln"] == "decode"


def test_decoder_layer_input_extraction_supports_old_and_new_cache_names():
    """Phase extraction must support both HF naming variants."""

    hidden = torch.randn(1, 1, 16)
    cache = _DummyCache(seq_len=3)

    phase_new = infer_phase_from_decoder_layer_inputs(
        args=(),
        kwargs={"hidden_states": hidden, "past_key_values": cache},
    )
    phase_old = infer_phase_from_decoder_layer_inputs(
        args=(),
        kwargs={"hidden_states": hidden, "past_key_value": cache},
    )

    assert phase_new == "decode"
    assert phase_old == "decode"


def test_attention_no_longer_writes_runtime_phase():
    """Guardrail: attention should consume phase context, not mutate it."""

    src = inspect.getsource(LlamaAttentionMXFP.forward)
    assert "set_active_phase(" not in src


def test_linear_mxfp_decode_uses_fp_snapshot_after_weight_replacement():
    """Decode path must use preserved FP weight even when prefill weights differ.

    Why this matters:
    GPTQ mutates source linear weights before module replacement. This test
    verifies the replacement seam restores the original FP snapshot for decode.
    """

    source = torch.nn.Linear(4, 3, bias=True)
    source_weight_quantized = torch.full_like(source.weight, 5.0)
    source_bias_quantized = torch.full_like(source.bias, -2.0)
    source.weight.data.copy_(source_weight_quantized)
    source.bias.data.copy_(source_bias_quantized)

    # Simulate snapshots captured before GPTQ in-place mutation.
    fp_weight = torch.full_like(source.weight, 2.0)
    fp_bias = torch.full_like(source.bias, 1.0)
    source._mase_decode_weight_fp = fp_weight.detach().clone()
    source._mase_decode_bias_fp = fp_bias.detach().clone()

    target = LinearMXFP(
        in_features=4,
        out_features=3,
        bias=True,
        config={"bypass": True},
    )
    target = weight_replacement(source, target)

    x = torch.randn(2, 4)

    set_active_phase("prefill")
    out_prefill = target(x)
    expected_prefill = torch.nn.functional.linear(
        x, source_weight_quantized, source_bias_quantized
    )
    assert torch.allclose(out_prefill, expected_prefill, atol=1e-6, rtol=0)

    set_active_phase("decode")
    out_decode = target(x)
    expected_decode = torch.nn.functional.linear(x, fp_weight, fp_bias)
    assert torch.allclose(out_decode, expected_decode, atol=1e-6, rtol=0)

    # Keep global test context deterministic for subsequent tests.
    set_active_phase("prefill")
