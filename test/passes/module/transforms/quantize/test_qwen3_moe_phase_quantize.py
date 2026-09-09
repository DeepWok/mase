import torch
import pytest
import transformers
import chop.passes.module.transforms.gptq.run as gptq_run

from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeExperts,
    Qwen3MoeSparseMoeBlock,
)

from chop.nn.quantized.modules.phase_config import (
    GPTQ_DECODE_EXPERT_DOWN_ATTR,
    GPTQ_DECODE_EXPERT_GATE_UP_ATTR,
)
from chop.nn.quantized.modules.phase_context import force_runtime_phase
from chop.nn.quantized.functional.matrix import plena_matrix_product
from chop.nn.quantized.modules.linear import LinearMXInt
from chop.nn.quantized.modules.qwen3_moe import (
    SUPPORTED_TRANSFORMERS_VERSION,
    Qwen3MoeAttentionMXInt,
    Qwen3MoeAttentionMXIntRotate,
    Qwen3MoeExpertsMXInt,
    Qwen3MoeSparseMoeBlockBF16Router,
    Qwen3MoeSparseMoeBlockMinifloat,
)
from chop.nn.quantized.modules.qwen3_moe.compat import require_qwen3_moe_fused_abi
from chop.passes.module.transforms.gptq.run import (
    _finalize_gptq_phase,
    _snapshot_fp_linear_weights,
)
from chop.passes.module.transforms.quantize.quantize import (
    quantize_module_transform_pass,
)
from chop.passes.module.transforms.quantize.rotation_search import (
    ALL_MATMUL_TYPES,
    _resolve_rotation_scope,
)
from chop.nn.quantizers import mxint_quantizer


def _config():
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


def _expert_qconfig():
    return {
        "prefill": {"bypass": True},
        "decode": {
            "weight_block_size": 4,
            "weight_width": 3,
            "data_in_block_size": 4,
            "data_in_width": 4,
        },
    }


def test_qwen3_moe_transformers_fused_abi_is_exactly_pinned():
    assert transformers.__version__ == SUPPORTED_TRANSFORMERS_VERSION == "5.5.0"
    require_qwen3_moe_fused_abi()


def test_untied_local_head_uses_decode_mx_matrix_and_bf16_logits():
    torch.manual_seed(20260725)
    model = Qwen3MoeForCausalLM(_config()).to(torch.bfloat16).eval()
    original = model.lm_head.weight.detach().clone()
    head_decode = {
        "weight_block_size": 8,
        "weight_width": 4,
        "data_in_block_size": 8,
        "data_in_width": 8,
        "output_format": "FP_E3M2",
        "matrix_mlen": 8,
    }
    model, _ = quantize_module_transform_pass(
        model,
        {
            "by": "regex_name",
            r"lm_head$": {
                "config": {
                    "name": "mxint",
                    "decode": head_decode,
                }
            },
        },
    )
    head = model.lm_head
    hidden = torch.randn(2, 1, 16, dtype=torch.bfloat16)

    assert isinstance(head, LinearMXInt)
    assert head._decode_weight_q.numel() == original.numel()
    torch.testing.assert_close(head.weight, original, rtol=0.0, atol=0.0)
    with force_runtime_phase("prefill"), torch.no_grad():
        prefill = head(hidden)
    torch.testing.assert_close(
        prefill,
        torch.nn.functional.linear(hidden, original),
        rtol=0.0,
        atol=0.0,
    )

    activation_q = mxint_quantizer(
        hidden,
        block_size=8,
        element_bits=8,
        block_dim=-1,
    )
    weight_q = mxint_quantizer(
        original,
        block_size=8,
        element_bits=4,
        block_dim=1,
    )
    expected = plena_matrix_product(
        activation_q,
        weight_q.transpose(-1, -2),
        head_decode,
    )
    with force_runtime_phase("decode"), torch.no_grad():
        decode = head(hidden)
        tied = head(torch.zeros_like(hidden))
    torch.testing.assert_close(decode, expected, rtol=0.0, atol=0.0)
    assert decode.dtype == torch.bfloat16
    assert int(tied.argmax(dim=-1)[0, 0].item()) == 0


def test_mase_matrix_oracle_mlen_partition_is_numerically_observable():
    """The sealed MLEN is part of accuracy, not only a timing parameter."""

    lhs = torch.tensor(
        [[0.69086325, -8.41099262, 1.58839154, 0.66403484]],
        dtype=torch.float32,
    )
    rhs = torch.tensor(
        [[0.68662047], [1.20273066], [6.97725391], [6.73511314]],
        dtype=torch.float32,
    )
    narrow = plena_matrix_product(
        lhs, rhs, {"output_format": "FP_E3M2", "matrix_mlen": 2}
    )
    wide = plena_matrix_product(
        lhs, rhs, {"output_format": "FP_E3M2", "matrix_mlen": 4}
    )

    torch.testing.assert_close(narrow, torch.tensor([[4.0]]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(wide, torch.tensor([[6.0]]), rtol=0.0, atol=0.0)


def test_selective_rotation_scope_excludes_fused_expert_tensors():
    model = Qwen3MoeForCausalLM(_config())
    eligible, scope = _resolve_rotation_scope(
        model, ALL_MATMUL_TYPES, explicitly_requested=False
    )

    assert set(eligible) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qk_matmul",
        "av_matmul",
        "kv_cache",
    }
    assert scope["architecture"] == "qwen3_moe_fused"
    assert scope["excluded_matmul_types"] == [
        "down_proj",
        "gate_proj",
        "up_proj",
    ]
    with pytest.raises(ValueError, match="does not support fused"):
        _resolve_rotation_scope(
            model, ("gate_proj",), explicitly_requested=True
        )


def test_fused_expert_banks_preserve_prefill_and_quantize_decode():
    torch.manual_seed(7)
    experts = Qwen3MoeExperts(_config())
    with torch.no_grad():
        experts.gate_up_proj.normal_(0.0, 0.1)
        experts.down_proj.normal_(0.0, 0.1)
    hidden = torch.randn(6, 16)
    indices = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]
    )
    scores = torch.full((6, 2), 0.5)
    reference = experts(hidden, indices, scores)

    quantized = Qwen3MoeExpertsMXInt.from_self(experts, _expert_qconfig())
    with force_runtime_phase("prefill"):
        prefill = quantized(hidden, indices, scores)
    with force_runtime_phase("decode"):
        decode = quantized(hidden, indices, scores)

    torch.testing.assert_close(prefill, reference)
    assert quantized._decode_gate_up_q.shape == experts.gate_up_proj.shape
    assert not torch.allclose(decode, reference)


def test_fused_expert_gptq_decode_handoff_restores_fp_source():
    model = Qwen3MoeForCausalLM(_config()).eval()
    experts = model.model.layers[0].mlp.experts
    fp_gate_up = experts.gate_up_proj.detach().clone()
    fp_down = experts.down_proj.detach().clone()

    _snapshot_fp_linear_weights(model.model.layers)
    with torch.no_grad():
        experts.gate_up_proj.add_(0.25)
        experts.down_proj.sub_(0.125)
    gptq_gate_up = experts.gate_up_proj.detach().clone()
    gptq_down = experts.down_proj.detach().clone()
    _finalize_gptq_phase(model, "decode")

    torch.testing.assert_close(experts.gate_up_proj, fp_gate_up)
    torch.testing.assert_close(experts.down_proj, fp_down)
    torch.testing.assert_close(
        getattr(experts, GPTQ_DECODE_EXPERT_GATE_UP_ATTR), gptq_gate_up
    )
    torch.testing.assert_close(
        getattr(experts, GPTQ_DECODE_EXPERT_DOWN_ATTR), gptq_down
    )
    replacement = Qwen3MoeExpertsMXInt.from_self(experts, _expert_qconfig())
    torch.testing.assert_close(replacement._decode_gate_up_q, gptq_gate_up)
    torch.testing.assert_close(replacement._decode_down_q, gptq_down)


def test_fused_expert_gptq_runs_end_to_end_on_cpu(monkeypatch):
    torch.manual_seed(20260725)
    model = Qwen3MoeForCausalLM(_config()).eval()
    experts = model.model.layers[0].mlp.experts
    fp_gate_up = experts.gate_up_proj.detach().clone()
    fp_down = experts.down_proj.detach().clone()
    samples = []
    for _ in range(4):
        input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
        samples.append((input_ids, input_ids.clone()))
    monkeypatch.setattr(gptq_run, "get_loaders", lambda *args, **kwargs: samples)

    gptq_run.run_gptq(
        model,
        {
            "model_name": "in-memory-tiny-qwen3-moe",
            "device": "cpu",
            "dataset": "in-memory",
            "nsamples": 4,
            "seqlen": 8,
            "format": "mxint",
            "weight_config": {"weight_block_size": 4, "weight_width": 4},
            "phase": "decode",
            "quantile_search": False,
            "clip_search_y": False,
            "cali_batch_size": 2,
            "min_expert_calibration_hits": 1,
        },
    )

    coverage = model._mase_gptq_expert_coverage
    assert len(coverage) == 1
    assert len(coverage[0]["gate_up"]) == experts.num_experts
    assert len(coverage[0]["down"]) == experts.num_experts
    assert all(record["gptq"] for record in coverage[0]["gate_up"])
    assert all(record["gptq"] for record in coverage[0]["down"])
    torch.testing.assert_close(experts.gate_up_proj, fp_gate_up)
    torch.testing.assert_close(experts.down_proj, fp_down)
    assert not torch.equal(
        getattr(experts, GPTQ_DECODE_EXPERT_GATE_UP_ATTR), fp_gate_up
    )
    assert not torch.equal(
        getattr(experts, GPTQ_DECODE_EXPERT_DOWN_ATTR), fp_down
    )
    assert model.config.use_cache is True
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
    assert torch.isfinite(logits).all()


def test_sparse_block_router_is_bf16_with_fp32_probabilities():
    block = Qwen3MoeSparseMoeBlockBF16Router.from_self(
        Qwen3MoeSparseMoeBlock(_config())
    )
    hidden = torch.randn(5, 16)
    probabilities, scores, indices = block.gate(hidden)
    assert block.gate.weight.dtype == torch.bfloat16
    assert probabilities.dtype == torch.float32
    assert scores.dtype == torch.float32
    assert indices.shape == (5, 2)
    torch.testing.assert_close(scores.sum(dim=-1), torch.ones(5))


def test_moe_attention_is_phase_aware_and_requires_tied_kv_precision():
    attention = Qwen3MoeAttention(_config(), layer_idx=0)
    rotate = Qwen3MoeAttentionMXIntRotate.from_self(
        attention,
        {
            "decode": {
                "qk_matmul": {"bypass": True, "rotate": False},
                "av_matmul": {"bypass": True, "rotate": True},
                "kv_cache": {"bypass": True, "rotate": False},
            }
        },
    )
    assert rotate.qk_use_rotate is False
    assert rotate.av_use_rotate is True
    assert rotate.kv_cache_use_rotate is False

    with pytest.raises(ValueError, match="identical K/V"):
        Qwen3MoeAttentionMXInt.from_self(
            attention,
            {
                "decode": {
                    "kv_cache": {
                        "key": {"data_in_block_size": 4, "data_in_width": 4},
                        "value": {"data_in_block_size": 4, "data_in_width": 3},
                    }
                }
            },
        )


def test_quantize_pass_replaces_packed_experts_and_hooks_moe_decoder():
    model = Qwen3MoeForCausalLM(_config()).eval()
    model, _ = quantize_module_transform_pass(
        model,
        {
            "by": "regex_name",
            r"model\.layers\.\d+\.mlp$": {
                "config": {"name": "bf16_router"}
            },
            r"model\.layers\.\d+\.mlp\.experts$": {
                "config": {"name": "mxint", **_expert_qconfig()}
            },
        },
    )
    layer = model.model.layers[0]
    assert isinstance(layer.mlp, Qwen3MoeSparseMoeBlockBF16Router)
    assert isinstance(layer.mlp.experts, Qwen3MoeExpertsMXInt)
    assert layer._mase_phase_hook_installed
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
    assert logits.shape == (1, 3, 64)
    assert torch.isfinite(logits).all()


def test_sparse_block_minifloat_keeps_router_safety_island():
    model = Qwen3MoeForCausalLM(_config()).eval()
    expert_config = _expert_qconfig()
    expert_config["decode"].update(
        {"output_format": "FP_E3M2", "matrix_mlen": 8}
    )
    model, _ = quantize_module_transform_pass(
        model,
        {
            "by": "regex_name",
            r"model\.layers\.\d+\.mlp$": {
                "config": {
                    "name": "minifloat",
                    "decode": {"format": "FP_E3M2"},
                }
            },
            r"model\.layers\.\d+\.mlp\.experts$": {
                "config": {"name": "mxint", **expert_config}
            },
        },
    )
    block = model.model.layers[0].mlp
    assert isinstance(block, Qwen3MoeSparseMoeBlockMinifloat)
    assert block.gate.weight.dtype == torch.bfloat16
    assert isinstance(block.experts, Qwen3MoeExpertsMXInt)
    assert block.experts.decode_config["output_format"] == "FP_E3M2"
    with force_runtime_phase("decode"), torch.no_grad():
        output = block(torch.randn(1, 3, 16))
    assert output.shape == (1, 3, 16)
    assert torch.isfinite(output).all()
