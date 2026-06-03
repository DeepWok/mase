"""Regression tests for phase-structured quantize pass wiring."""

from __future__ import annotations

from copy import deepcopy
import sys
import types
from pathlib import Path

import torch
import pytest
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
if "chop" not in sys.modules:
    repo_root = next(
        p for p in Path(__file__).resolve().parents if (p / "src/chop").exists()
    )
    chop_stub = types.ModuleType("chop")
    chop_stub.__path__ = [str(repo_root / "src/chop")]
    sys.modules["chop"] = chop_stub

from chop.nn.quantized.modules.linear import LinearMXFP
from chop.passes.module.transforms.quantize.quantize import (
    quantize_module_transform_pass,
    _install_llama_phase_context_pre_hooks,
    _infer_llama_decode_policy_from_quantized_modules,
)


class _TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.fc2 = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_quantize_pass_accepts_phase_config_for_mxfp_linear():
    """Phase config should be forwarded to phase-aware linear module."""

    model = _TinyMLP()
    pass_args = {
        "by": "name",
        "fc1": {
            "config": {
                "name": "mxfp",
                "decode_policy": "fp_only",
                "prefill": {"bypass": True},
                "decode": {"bypass": True},
            }
        },
    }
    pass_args_before = deepcopy(pass_args)

    quantized_model, _ = quantize_module_transform_pass(model, pass_args)
    assert isinstance(quantized_model.fc1, LinearMXFP)
    assert quantized_model.fc1.phase_config["decode_policy"] == "fp_only"
    assert quantized_model.fc1.phase_config["prefill"]["bypass"] is True
    assert quantized_model.fc1.phase_config["decode"]["bypass"] is True

    # Caller-owned config should remain unchanged after the pass.
    assert pass_args == pass_args_before


def test_quantize_pass_accepts_quantized_decode_policy():
    """`decode_policy=quantized` should pass parsing and instantiation."""

    model = _TinyMLP()
    pass_args = {
        "by": "name",
        "fc1": {
            "config": {
                "name": "mxfp",
                "decode_policy": "quantized",
                "prefill": {"bypass": True},
                "decode": {
                    "bypass": False,
                    "data_in_block_size": 16,
                    "data_in_exponent_width": 4,
                    "data_in_frac_width": 3,
                    "weight_block_size": 16,
                    "weight_exponent_width": 4,
                    "weight_frac_width": 3,
                },
            }
        },
    }
    quantized_model, _ = quantize_module_transform_pass(model, pass_args)
    assert isinstance(quantized_model.fc1, LinearMXFP)
    assert quantized_model.fc1.decode_policy == "quantized"


def test_llama_phase_pre_hook_installation_is_gated_and_idempotent():
    """Hooks should install only for quantized Llama runs and only once."""

    class LlamaAttentionMXFP(torch.nn.Module):
        def __init__(self, decode_policy: str = "fp_only"):
            super().__init__()
            self.decode_policy = decode_policy

        def forward(self, x):
            return x

    class LinearMXFP(torch.nn.Module):
        def __init__(self, decode_policy: str = "fp_only"):
            super().__init__()
            self.decode_policy = decode_policy

        def forward(self, x):
            return x

    class TinyNetwork(torch.nn.Module):
        def __init__(
            self,
            with_quantized_marker: bool,
            marker_class: type[torch.nn.Module] = LlamaAttentionMXFP,
            decode_policy: str = "fp_only",
        ):
            super().__init__()
            cfg = LlamaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
            )
            self.layer = LlamaDecoderLayer(cfg, layer_idx=0)
            if with_quantized_marker:
                self.quant_marker = marker_class(decode_policy=decode_policy)

    net_without_marker = TinyNetwork(with_quantized_marker=False)
    before_no_marker = len(net_without_marker.layer._forward_pre_hooks)
    _install_llama_phase_context_pre_hooks(net_without_marker)
    assert len(net_without_marker.layer._forward_pre_hooks) == before_no_marker

    net_with_marker = TinyNetwork(with_quantized_marker=True)
    before_with_marker = len(net_with_marker.layer._forward_pre_hooks)
    _install_llama_phase_context_pre_hooks(net_with_marker)
    after_first = len(net_with_marker.layer._forward_pre_hooks)
    _install_llama_phase_context_pre_hooks(net_with_marker)
    after_second = len(net_with_marker.layer._forward_pre_hooks)

    assert after_first == before_with_marker + 1
    assert after_second == after_first

    # Linear-only quantization must also install the hook, because runtime
    # phase/decode policy is consumed inside quantized Linear forward paths.
    net_with_linear_marker = TinyNetwork(
        with_quantized_marker=True, marker_class=LinearMXFP
    )
    before_linear_marker = len(net_with_linear_marker.layer._forward_pre_hooks)
    _install_llama_phase_context_pre_hooks(net_with_linear_marker)
    assert len(net_with_linear_marker.layer._forward_pre_hooks) == (
        before_linear_marker + 1
    )


def test_decode_policy_inference_fails_fast_on_mixed_llama_policies():
    """Mixed decode policies must raise instead of silently choosing one."""

    class LlamaAttentionMXFP(torch.nn.Module):
        def __init__(self, decode_policy: str):
            super().__init__()
            self.decode_policy = decode_policy

    class TinyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = LlamaAttentionMXFP("fp_only")
            self.b = LlamaAttentionMXFP("quantized")

    with pytest.raises(ValueError, match="Mixed decode policies"):
        _infer_llama_decode_policy_from_quantized_modules(TinyNetwork())
