from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)

from chop.nn.quantized.functional.matrix import plena_matrix_product
from chop.nn.quantized.modules.phase_context import force_runtime_phase
from chop.nn.quantized.modules.qwen3_moe import (
    Qwen3MoeTopKRouterBF16,
    Qwen3MoeTopKRouterE4M3,
    Qwen3MoeTopKRouterE5M2,
    Qwen3MoeTopKRouterMX,
    Qwen3MoeTopKRouterMXInt8,
    router_phase_config,
)
from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.passes.module.transforms.quantize.quantize import (
    quantize_module_transform_pass,
)


def _config() -> Qwen3MoeConfig:
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    return config


def _mlen_config() -> Qwen3MoeConfig:
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=2048,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=512,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    return config


def _manual_quantize(value: torch.Tensor, token: str, block_dim: int):
    if token == "MXINT8":
        return mxint_quantizer(
            value, block_size=8, element_bits=8, block_dim=block_dim
        )
    exponent, fraction = token.removeprefix("E").split("M")
    return mxfp_quantizer(
        value,
        block_size=8,
        element_exp_bits=int(exponent),
        element_frac_bits=int(fraction),
        block_dim=block_dim,
    )


def _manual_route(logits: torch.Tensor, top_k: int):
    probabilities = F.softmax(logits.to(torch.bfloat16), dtype=torch.float32, dim=-1)
    scores, indices = torch.topk(probabilities, top_k, dim=-1, sorted=True)
    scores = scores / scores.sum(dim=-1, keepdim=True)
    return probabilities, scores, indices


@pytest.mark.parametrize(
    ("token", "router_cls"),
    (
        ("MXINT8", Qwen3MoeTopKRouterMXInt8),
        ("E4M3", Qwen3MoeTopKRouterE4M3),
        ("E5M2", Qwen3MoeTopKRouterE5M2),
    ),
)
def test_router_mx_cpu_matches_explicit_matrix_and_fp32_route(token, router_cls):
    torch.manual_seed(20260820)
    source = Qwen3MoeSparseMoeBlock(_config()).gate
    source.weight.data.copy_(
        torch.linspace(-0.75, 0.75, source.weight.numel()).reshape_as(source.weight)
    )
    config = router_phase_config(token, token, matrix_mlen=8)
    router = router_cls.from_self(source, config)
    hidden = torch.linspace(-1.0, 1.0, 5 * 16).reshape(5, 16)

    activation_q = _manual_quantize(hidden.to(torch.bfloat16), token, -1)
    weight_q = _manual_quantize(source.weight.to(torch.bfloat16), token, 1)
    logits = plena_matrix_product(
        activation_q,
        weight_q.transpose(-1, -2),
        {"matrix_mlen": 8, "output_format": "BF16"},
    ).to(torch.bfloat16)
    expected = _manual_route(logits, top_k=2)

    with force_runtime_phase("decode"), torch.no_grad():
        observed = router(hidden)
    for actual, reference in zip(observed, expected):
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    assert observed[0].dtype == torch.float32
    assert observed[1].dtype == torch.float32
    assert router.router_precision_contract["publication_rankable"] is False
    assert router.router_precision_contract["selection_eligible"] is False


def test_router_mx_prefill_has_exact_bf16_baseline_ancestry():
    torch.manual_seed(9)
    source = Qwen3MoeSparseMoeBlock(_config()).gate
    baseline = Qwen3MoeTopKRouterBF16.from_self(source)
    variant = Qwen3MoeTopKRouterMX.from_self(
        source,
        router_phase_config("E4M3", "E5M2", matrix_mlen=16),
    )
    hidden = torch.randn(7, 16)
    with force_runtime_phase("prefill"), torch.no_grad():
        baseline_output = baseline(hidden)
        variant_output = variant(hidden)
    for actual, reference in zip(variant_output, baseline_output):
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)


def test_router_matrix_oracle_exposes_mlen_1024_vs_2048_partition_boundary():
    torch.manual_seed(17)
    source = Qwen3MoeSparseMoeBlock(_mlen_config()).gate
    source.weight.data.normal_(mean=0.0, std=0.01)
    hidden = torch.randn(5, 2048) * 0.1
    split = Qwen3MoeTopKRouterE4M3.from_self(
        source,
        router_phase_config("E4M3", "E4M3", matrix_mlen=1024),
    )
    whole = Qwen3MoeTopKRouterE4M3.from_self(
        source,
        router_phase_config("E4M3", "E4M3", matrix_mlen=2048),
    )
    with force_runtime_phase("decode"), torch.no_grad():
        split_probabilities, _, _ = split(hidden)
        whole_probabilities, _, _ = whole(hidden)
    assert not torch.equal(split_probabilities, whole_probabilities)
    assert split.router_precision_contract["matrix_arithmetic_chain"] == [
        "per_mlen_fp32_matmul",
        "bf16_partial_rounding",
        "truncate_partial_to_signed_fixed16_16",
        "signed_fixed16_16_wrap_across_partitions",
        "final_bf16_writeout",
    ]


def test_router_route_agreement_measurement_is_deterministic_on_cpu():
    torch.manual_seed(112)
    source = Qwen3MoeSparseMoeBlock(_config()).gate
    baseline = Qwen3MoeTopKRouterBF16.from_self(source)
    variant = Qwen3MoeTopKRouterE5M2.from_self(
        source,
        router_phase_config("E5M2", "E5M2", matrix_mlen=8),
    )
    hidden = torch.randn(13, 16)

    def measure():
        with force_runtime_phase("decode"), torch.no_grad():
            base_prob, _, base_idx = baseline(hidden)
            test_prob, _, test_idx = variant(hidden)
        ordered = (base_idx == test_idx).all(dim=-1)
        set_equal = torch.tensor(
            [
                torch.equal(torch.sort(left).values, torch.sort(right).values)
                for left, right in zip(base_idx, test_idx)
            ],
            dtype=torch.bool,
        )
        return {
            "tokens": int(hidden.shape[0]),
            "ordered_matches": int(ordered.sum()),
            "set_matches": int(set_equal.sum()),
            "probability_l1_sum": float((base_prob - test_prob).abs().sum()),
            "probability_linf": float((base_prob - test_prob).abs().max()),
        }

    first = measure()
    second = measure()
    assert first == second
    assert 0 <= first["ordered_matches"] <= first["set_matches"] <= 13
    assert first["probability_l1_sum"] >= 0.0
    assert first["probability_linf"] >= 0.0


def test_quantize_pass_installs_router_variant_without_touching_profile_space():
    model = Qwen3MoeForCausalLM(_config()).eval()
    model, _ = quantize_module_transform_pass(
        model,
        {
            "by": "regex_name",
            r"model\.layers\.\d+\.mlp\.gate$": {
                "config": {
                    "name": "e4m3",
                    **router_phase_config("E4M3", "E4M3", matrix_mlen=8),
                }
            },
        },
    )
    gate = model.model.layers[0].mlp.gate
    assert isinstance(gate, Qwen3MoeTopKRouterE4M3)
    assert gate.router_weight_format == gate.router_activation_format == "E4M3"
    with force_runtime_phase("decode"), torch.no_grad():
        probability, score, index = gate(torch.randn(3, 16))
    assert probability.shape == (3, 4)
    assert score.shape == index.shape == (3, 2)


@pytest.mark.parametrize("token", ("MXINT4", "E3M4", "BF16"))
def test_router_lane_rejects_unplanned_formats(token):
    source = Qwen3MoeSparseMoeBlock(_config()).gate
    with pytest.raises(ValueError, match="must be one of"):
        Qwen3MoeTopKRouterMX.from_self(
            source,
            router_phase_config(token, "MXINT8", matrix_mlen=8),
        )
