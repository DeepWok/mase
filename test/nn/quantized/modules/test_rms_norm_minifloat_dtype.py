"""The minifloat RMSNorm modules must return tensors in the input dtype.

Module replacement constructs the quantised RMSNorm before the model is moved
to its working dtype, so ``weight`` can still be float32 while the activations
are bfloat16. ``float32 * bfloat16`` promotes to float32, and the following
linear then fails on a dtype mismatch. The modules cast the product back to the
input dtype; these tests pin that contract.
"""

import pytest
import torch
from transformers import LlamaConfig, Qwen3Config

from chop.nn.quantized.modules.llama.rms_norm import LlamaRMSNormMinifloat
from chop.nn.quantized.modules.qwen3.rms_norm import Qwen3RMSNormMinifloat

HIDDEN = 16
EPS = 1e-6

QUANTISED = {
    "weight_exponent_width": 4,
    "weight_frac_width": 3,
    "data_in_exponent_width": 4,
    "data_in_frac_width": 3,
}
BYPASS = {"bypass": True}

CASES = [
    (LlamaRMSNormMinifloat, LlamaConfig(hidden_size=HIDDEN, rms_norm_eps=EPS)),
    (Qwen3RMSNormMinifloat, Qwen3Config(hidden_size=HIDDEN, rms_norm_eps=EPS)),
]


def _reference(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    return (weight * normed.to(x.dtype)).to(x.dtype)


@pytest.mark.parametrize("cls,config", CASES, ids=["llama", "qwen3"])
@pytest.mark.parametrize("q_config", [QUANTISED, BYPASS], ids=["quantised", "bypass"])
def test_bf16_input_with_fp32_weight_returns_bf16(cls, config, q_config):
    module = cls(config=config, q_config=q_config)
    assert module.weight.dtype == torch.float32

    hidden = torch.randn(3, HIDDEN).to(torch.bfloat16)
    out = module(hidden)

    assert out.dtype == torch.bfloat16
    assert out.shape == hidden.shape
    if q_config is BYPASS:
        torch.testing.assert_close(out, _reference(module.weight, hidden))


@pytest.mark.parametrize("cls,config", CASES, ids=["llama", "qwen3"])
def test_fp32_input_stays_fp32(cls, config):
    module = cls(config=config, q_config=BYPASS)
    hidden = torch.randn(3, HIDDEN)
    out = module(hidden)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _reference(module.weight, hidden))
