"""Decode-only phase-split quantisation tests (PLENA disaggregated serving).

Contract under test:
- Prefill stays exactly FP (the unquantised prefill chip).
- Decode consumes quantised weight banks / activations / KV cache.
- KV handoff: prefill KV writes land in the decode chip's MX format.
- GPTQ ``phase="decode"`` results become the decode weight bank while FP
  weights are restored for prefill.
- Legacy flat configs keep quantising both phases identically (no silent
  behaviour change, zero extra memory).

All tests are small, deterministic, and CPU-only (no downloads).
"""

import copy

import pytest
import torch
import torch.nn as nn

from chop.nn.quantized.modules.linear import (
    LinearMXFP,
    LinearMXInt,
    RotateMXIntLinear,
)
from chop.nn.quantized.modules.phase_config import (
    normalize_phase_q_config,
    resolve_module_phase_config,
    resolve_stage_config,
)
from chop.nn.quantized.modules.phase_context import (
    force_runtime_phase,
    get_runtime_phase,
    infer_runtime_phase_from_hidden_and_cache,
    set_runtime_phase,
)
from chop.nn.quantizers import mxint_quantizer
from chop.passes.module.module_modify_helper import weight_replacement
from chop.passes.module.transforms.gptq.run import (
    _finalize_gptq_phase,
    _snapshot_fp_linear_weights,
)
from chop.passes.module.transforms.quantize.quantize import (
    quantize_module_transform_pass,
)

MXINT_LINEAR_DECODE = {
    "weight_block_size": 8,
    "weight_width": 4,
    "data_in_block_size": 8,
    "data_in_width": 8,
}


@pytest.fixture(autouse=True)
def _reset_phase():
    set_runtime_phase("prefill")
    yield
    set_runtime_phase("prefill")


# ---------------------------------------------------------------------------
# phase_config normalization
# ---------------------------------------------------------------------------


def test_legacy_flat_config_quantises_both_phases():
    n = normalize_phase_q_config(dict(MXINT_LINEAR_DECODE))
    assert n["decode_policy"] == "quantized"
    assert n["prefill"] == MXINT_LINEAR_DECODE
    assert n["decode"] == MXINT_LINEAR_DECODE


def test_decode_only_shorthand_defaults_prefill_to_bypass():
    n = normalize_phase_q_config({"decode": dict(MXINT_LINEAR_DECODE)})
    assert n["prefill"] == {"bypass": True}
    assert n["decode_policy"] == "quantized"
    assert resolve_module_phase_config(n, "prefill") == {"bypass": True}
    assert resolve_module_phase_config(n, "decode") == MXINT_LINEAR_DECODE


def test_fp_only_policy_force_bypasses_decode():
    n = normalize_phase_q_config(
        {
            "prefill": dict(MXINT_LINEAR_DECODE),
            "decode": dict(MXINT_LINEAR_DECODE),
            "decode_policy": "fp_only",
        }
    )
    assert resolve_module_phase_config(n, "decode") == {"bypass": True}
    assert resolve_module_phase_config(n, "prefill") == MXINT_LINEAR_DECODE


def test_decode_policy_inferred_fp_only_for_bypass_decode():
    n = normalize_phase_q_config({"prefill": dict(MXINT_LINEAR_DECODE), "decode": {"bypass": True}})
    assert n["decode_policy"] == "fp_only"


def test_normalization_is_idempotent():
    cfg = {"prefill": {"bypass": True}, "decode": dict(MXINT_LINEAR_DECODE)}
    once = normalize_phase_q_config(cfg)
    assert normalize_phase_q_config(once) == once


def test_decode_quantization_rejects_legacy_mxint_route():
    model = nn.Sequential(nn.Linear(8, 8, bias=False))
    pass_args = {
        "by": "type",
        "linear": {
            "config": {
                "name": "mxint_hardware",
                "prefill": {"bypass": True},
                "decode": dict(MXINT_LINEAR_DECODE),
            }
        },
    }
    with pytest.raises(
        ValueError,
        match=r"decode quantization must select 'mxint'.*mxint\.fake",
    ):
        quantize_module_transform_pass(model, pass_args)


def test_kv_handoff_decode_format_mirrors_decode_kv_into_prefill():
    kv = {"data_in_block_size": 8, "data_in_width": 4}
    n = normalize_phase_q_config(
        {
            "kv_cache_handoff": "decode_format",
            "prefill": {"bypass": True},
            "decode": {"kv_cache": dict(kv)},
        }
    )
    assert resolve_stage_config(n, "prefill", "kv_cache") == kv
    # every other prefill stage stays bypassed
    assert resolve_stage_config(n, "prefill", "qk_matmul") == {"bypass": True}


def test_kv_handoff_defaults_to_fp_and_keeps_prefill_kv_unquantised():
    """The prefill cache crosses the chip boundary in the prefill dtype.

    Decode admission is what quantizes it, so the default handoff must leave
    prefill KV untouched.
    """
    n = normalize_phase_q_config(
        {
            "prefill": {"bypass": True},
            "decode": {"kv_cache": {"data_in_block_size": 8, "data_in_width": 4}},
        }
    )
    assert resolve_stage_config(n, "prefill", "kv_cache") == {"bypass": True}


def test_explicit_prefill_kv_beats_handoff_rule():
    explicit = {"data_in_block_size": 16, "data_in_width": 8}
    n = normalize_phase_q_config(
        {
            "prefill": {"bypass": True, "kv_cache": dict(explicit)},
            "decode": {"kv_cache": {"data_in_block_size": 8, "data_in_width": 4}},
        }
    )
    assert resolve_stage_config(n, "prefill", "kv_cache") == explicit


# ---------------------------------------------------------------------------
# phase_context
# ---------------------------------------------------------------------------


class _DummyCache:
    def __init__(self, seq_len: int):
        self._seq_len = seq_len

    def get_seq_length(self) -> int:
        return self._seq_len


def test_phase_inference_from_cache_semantics():
    hidden_prefill = torch.randn(1, 8, 16)
    hidden_decode = torch.randn(1, 1, 16)
    assert infer_runtime_phase_from_hidden_and_cache(hidden_prefill, None) == "prefill"
    assert (
        infer_runtime_phase_from_hidden_and_cache(hidden_prefill, _DummyCache(0))
        == "prefill"
    )
    # Single-token prompt: cache_position starts at 0 = prefill.
    assert (
        infer_runtime_phase_from_hidden_and_cache(
            hidden_decode, _DummyCache(0), cache_position=torch.tensor([0])
        )
        == "prefill"
    )
    # Decode step: single token appended past existing cache content.
    assert (
        infer_runtime_phase_from_hidden_and_cache(
            hidden_decode, _DummyCache(32), cache_position=torch.tensor([32])
        )
        == "decode"
    )
    assert (
        infer_runtime_phase_from_hidden_and_cache(hidden_decode, _DummyCache(32))
        == "decode"
    )
    # Multi-token forward is always prefill, even when earlier layers have
    # already written this prompt's KV into the cache.
    assert (
        infer_runtime_phase_from_hidden_and_cache(hidden_prefill, _DummyCache(32))
        == "prefill"
    )


def test_cached_prefill_classifies_every_layer_as_prefill():
    """During a cached prefill, layer 0 fills the cache before later layers
    run — their hooks must still classify the forward as prefill."""
    hidden = torch.randn(1, 8, 16)
    # Layer 0 view: empty cache. Layer 1+ view: cache already holds the
    # prompt KV written by layer 0. Both must resolve to prefill.
    assert infer_runtime_phase_from_hidden_and_cache(hidden, _DummyCache(0)) == "prefill"
    assert infer_runtime_phase_from_hidden_and_cache(hidden, _DummyCache(8)) == "prefill"


def test_force_runtime_phase_overrides_hook_writes():
    set_runtime_phase("prefill")
    with force_runtime_phase("decode"):
        assert get_runtime_phase() == "decode"
        # a decoder-layer hook firing mid-eval cannot break the override
        set_runtime_phase("prefill")
        assert get_runtime_phase() == "decode"
    assert get_runtime_phase() == "prefill"


# ---------------------------------------------------------------------------
# phase-aware linear banks
# ---------------------------------------------------------------------------


def _decode_only_cfg():
    return {"prefill": {"bypass": True}, "decode": dict(MXINT_LINEAR_DECODE)}


def test_linear_decode_only_prefill_fp_decode_quantised():
    src = nn.Linear(16, 8, bias=True)
    lin = LinearMXInt(16, 8, bias=True, config=_decode_only_cfg())
    lin.load_state_dict(src.state_dict(), strict=False)

    x = torch.randn(2, 16)
    set_runtime_phase("prefill")
    assert torch.allclose(
        lin(x), nn.functional.linear(x, src.weight, src.bias), atol=1e-6
    )
    assert torch.equal(lin.weight, src.weight)

    set_runtime_phase("decode")
    w_q = mxint_quantizer(src.weight, block_size=8, element_bits=4, block_dim=1)
    x_q = mxint_quantizer(x, block_size=8, element_bits=8, block_dim=-1)
    assert torch.allclose(lin(x), nn.functional.linear(x_q, w_q, src.bias), atol=1e-6)


def test_linear_legacy_flat_single_bank():
    src = nn.Linear(16, 8, bias=True)
    lin = LinearMXInt(16, 8, bias=True, config=dict(MXINT_LINEAR_DECODE))
    lin.load_state_dict(src.state_dict(), strict=False)
    assert lin.shared_phase_banks
    assert lin._decode_weight_q.numel() == 0
    w_q = mxint_quantizer(src.weight, block_size=8, element_bits=4, block_dim=1)
    assert torch.allclose(lin.weight, w_q, atol=1e-6)
    x = torch.randn(2, 16)
    set_runtime_phase("prefill")
    o1 = lin(x)
    set_runtime_phase("decode")
    assert torch.equal(lin(x), o1)


def test_linear_gptq_decode_stash_adopted_at_replacement():
    src = nn.Linear(16, 8, bias=True)
    gptq_w = torch.full_like(src.weight, 0.5)
    src._mase_gptq_weight_decode = gptq_w.clone()
    cfg = {
        "prefill": {"bypass": True},
        "decode": {"gptq": True, "data_in_block_size": 8, "data_in_width": 8},
    }
    tgt = LinearMXInt(16, 8, bias=True, config=cfg)
    tgt = weight_replacement(src, tgt)
    assert torch.equal(tgt.weight, src.weight)
    assert torch.equal(tgt._decode_weight_q, gptq_w)

    x = torch.randn(2, 16)
    set_runtime_phase("decode")
    x_q = mxint_quantizer(x, block_size=8, element_bits=8, block_dim=-1)
    assert torch.allclose(
        tgt(x), nn.functional.linear(x_q, gptq_w, src.bias), atol=1e-6
    )


def test_linear_prefill_side_fp_only_flow():
    """Inverse deployment: quantised prefill + FP-snapshot decode."""
    src = nn.Linear(16, 8, bias=True)
    cfg = {
        "prefill": {"weight_block_size": 8, "weight_width": 4},
        "decode": {"bypass": True},
        "decode_policy": "fp_only",
    }
    tgt = LinearMXInt(16, 8, bias=True, config=cfg)
    tgt = weight_replacement(src, tgt)
    assert tgt._decode_weight_fp.numel() > 0

    x = torch.randn(2, 16)
    set_runtime_phase("decode")
    assert torch.allclose(
        tgt(x), nn.functional.linear(x, src.weight, src.bias), atol=1e-6
    )
    set_runtime_phase("prefill")
    w_q = mxint_quantizer(src.weight, block_size=8, element_bits=4, block_dim=1)
    assert torch.allclose(tgt(x), nn.functional.linear(x, w_q, src.bias), atol=1e-6)


def test_rotation_search_swap_shares_parameters_and_banks():
    src = nn.Linear(16, 8, bias=True)
    lin = LinearMXInt(16, 8, bias=True, config=_decode_only_cfg())
    lin.load_state_dict(src.state_dict(), strict=False)
    rot = RotateMXIntLinear.from_linear(lin, lin.config)
    assert rot.weight is lin.weight
    assert rot._decode_weight_q is lin._decode_weight_q
    back = LinearMXInt.from_linear(rot, rot.config)
    assert back._decode_weight_q is lin._decode_weight_q


def test_rotate_linear_decode_only_phases():
    src = nn.Linear(64, 32, bias=False)
    rot = RotateMXIntLinear(64, 32, bias=False, config=_decode_only_cfg())
    rot.load_state_dict(src.state_dict(), strict=False)
    x = torch.randn(2, 64)
    set_runtime_phase("prefill")
    assert torch.allclose(
        rot(x), nn.functional.linear(x, src.weight), atol=1e-6
    ), "bypassed prefill must skip the rotation round-trip entirely"
    set_runtime_phase("decode")
    plain = LinearMXInt(64, 32, bias=False, config=_decode_only_cfg())
    plain.load_state_dict(src.state_dict(), strict=False)
    assert not torch.allclose(rot(x), plain(x)), "rotation must alter quant error"


def test_linear_mxfp_decode_only():
    src = nn.Linear(16, 8, bias=True)
    cfg = {
        "prefill": {"bypass": True},
        "decode": {
            "weight_block_size": 8,
            "weight_exponent_width": 4,
            "weight_frac_width": 3,
            "data_in_block_size": 8,
            "data_in_exponent_width": 4,
            "data_in_frac_width": 3,
        },
    }
    lin = LinearMXFP(16, 8, bias=True, config=cfg)
    lin.load_state_dict(src.state_dict(), strict=False)
    x = torch.randn(2, 16)
    set_runtime_phase("prefill")
    out_p = lin(x)
    assert torch.allclose(out_p, nn.functional.linear(x, src.weight, src.bias), atol=1e-6)
    set_runtime_phase("decode")
    assert lin._decode_weight_q.numel() > 0
    assert not torch.allclose(lin(x), out_p)


# ---------------------------------------------------------------------------
# GPTQ phase plumbing
# ---------------------------------------------------------------------------


class _TinyDecoderNet(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()

        def layer():
            l = nn.Module()
            l.self_attn = nn.Module()
            l.self_attn.q_proj = nn.Linear(8, 8)
            return l

        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer() for _ in range(n_layers)])


def test_gptq_phase_decode_stashes_and_restores_fp():
    net = _TinyDecoderNet()
    fp = {
        id(l): l.self_attn.q_proj.weight.detach().clone() for l in net.model.layers
    }
    _snapshot_fp_linear_weights(net.model.layers)
    for l in net.model.layers:  # simulate GPTQ in-place mutation
        l.self_attn.q_proj.weight.data.fill_(7.0)
    _finalize_gptq_phase(net, "decode")
    for l in net.model.layers:
        q = l.self_attn.q_proj
        assert torch.equal(q.weight, fp[id(l)])
        assert torch.all(q._mase_gptq_weight_decode == 7.0)
        assert not hasattr(q, "_mase_fp_weight")


def test_gptq_phase_prefill_keeps_gptq_and_snapshots_fp():
    net = _TinyDecoderNet()
    _snapshot_fp_linear_weights(net.model.layers)
    for l in net.model.layers:
        l.self_attn.q_proj.weight.data.fill_(3.0)
    _finalize_gptq_phase(net, "prefill")
    for l in net.model.layers:
        q = l.self_attn.q_proj
        assert torch.all(q.weight == 3.0)
        assert q._mase_decode_weight_fp is not None


# ---------------------------------------------------------------------------
# end-to-end: tiny Llama through the quantize pass
# ---------------------------------------------------------------------------


def _tiny_llama():
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg).eval()


def _cache_keys(cache):
    """Read layer-0 keys from either DynamicCache layout."""
    return cache.layers[0].keys if hasattr(cache, "layers") else cache.key_cache[0]


def _llama_decode_only_pass_args(kv_cache_handoff: str | None = None):
    mx = {"data_in_block_size": 16, "data_in_width": 4}
    attention = {
        "name": "mxint",
        "prefill": {"bypass": True},
        "decode": {
            "qk_matmul": dict(mx),
            "av_matmul": dict(mx),
            "rope": {"bypass": True},
            "softmax": {"bypass": True},
            "kv_cache": {"data_in_block_size": 16, "data_in_width": 4},
        },
    }
    if kv_cache_handoff is not None:
        attention["kv_cache_handoff"] = kv_cache_handoff
    return {
        "by": "regex_name",
        r"model\.layers\.\d+\.self_attn$": {"config": attention},
        r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        r"|mlp\.(gate_proj|up_proj|down_proj))$": {
            "config": {
                "name": "mxint",
                "prefill": {"bypass": True},
                "decode": {
                    "weight_block_size": 16,
                    "weight_width": 4,
                    "data_in_block_size": 16,
                    "data_in_width": 8,
                },
            }
        },
    }


def test_llama_decode_only_end_to_end():
    ref = _tiny_llama()
    model = copy.deepcopy(ref)
    model, _ = quantize_module_transform_pass(model, _llama_decode_only_pass_args())

    n_hooks = sum(
        1 for m in model.modules() if getattr(m, "_mase_phase_hook_installed", False)
    )
    assert n_hooks == 2

    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        out_q = model(ids, use_cache=False).logits
        out_ref = ref(ids, use_cache=False).logits
    # Cache-free forward = pure prefill = must be EXACTLY the FP reference.
    assert torch.allclose(out_q, out_ref, atol=1e-6)

    # Prefill with cache: the prompt cache crosses the boundary unquantised.
    from transformers import DynamicCache

    with torch.no_grad():
        cache = DynamicCache()
        model(ids, past_key_values=cache, use_cache=True)
    k0 = _cache_keys(cache)
    assert not torch.equal(k0, torch.zeros_like(k0))
    k0_requant = mxint_quantizer(k0, block_size=16, element_bits=4, block_dim=-1)
    assert not torch.allclose(k0, k0_requant, atol=1e-6)

    # Greedy decode runs through the quantised decode path without error.
    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, do_sample=False)

    # Per-phase dispatch on a replaced projection.
    lin0 = model.model.layers[0].self_attn.q_proj
    x = torch.randn(1, 64)
    set_runtime_phase("prefill")
    o_p = lin0(x)
    set_runtime_phase("decode")
    assert not torch.allclose(lin0(x), o_p)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prefill_stays_bit_identical_in_task_dtype(dtype):
    """The unquantised prefill chip must run in the task's own dtype.

    A bf16 model keeps bf16 prefill, an fp16 model keeps fp16 prefill —
    logits are bit-identical to the unquantised reference, and the decode
    weight banks are stored as fake-quant values in the same task dtype.
    """
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    ref = LlamaForCausalLM(cfg).to(dtype).eval()
    model = copy.deepcopy(ref)
    model, _ = quantize_module_transform_pass(model, _llama_decode_only_pass_args())

    lin = model.model.layers[0].self_attn.q_proj
    assert lin.weight.dtype == dtype
    assert torch.equal(lin.weight, ref.model.layers[0].self_attn.q_proj.weight)
    assert lin._decode_weight_q.dtype == dtype

    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        out_q = model(ids, use_cache=False).logits
        out_ref = ref(ids, use_cache=False).logits
    assert out_q.dtype == dtype
    assert torch.equal(out_q, out_ref)


def test_llama_cached_prefill_is_bit_identical_fp():
    """The real generate() prefill runs WITH a KV cache. Every layer must
    still execute the FP prefill path, and the prompt's own attention must
    use FP K/V (only the cache copy is quantised), so the prefill logits are
    bit-identical to the unquantised reference."""
    ref = _tiny_llama()
    model = copy.deepcopy(ref)
    model, _ = quantize_module_transform_pass(model, _llama_decode_only_pass_args())

    from transformers import DynamicCache

    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        out_q = model(ids, past_key_values=DynamicCache(), use_cache=True).logits
        out_ref = ref(ids, use_cache=False).logits
    assert torch.equal(out_q, out_ref)

    # The prompt cache stays in the prefill dtype under the default handoff.
    with torch.no_grad():
        cache = DynamicCache()
        model(ids, past_key_values=cache, use_cache=True)
    k0 = _cache_keys(cache)
    assert not torch.equal(k0, torch.zeros_like(k0))
    k0_requant = mxint_quantizer(k0, block_size=16, element_bits=4, block_dim=-1)
    assert not torch.allclose(k0, k0_requant, atol=1e-6)


def test_kv_handoff_decode_format_writes_quantised_prefill_cache():
    """Opting into decode_format quantises the prompt cache on write.

    This models a prefill chip that emits KV already in the decode chip's MX
    format, instead of quantising at decode admission.
    """
    ref = _tiny_llama()
    model = copy.deepcopy(ref)
    model, _ = quantize_module_transform_pass(
        model, _llama_decode_only_pass_args(kv_cache_handoff="decode_format")
    )

    from transformers import DynamicCache

    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        cache = DynamicCache()
        model(ids, past_key_values=cache, use_cache=True)
    k0 = _cache_keys(cache)
    assert not torch.equal(k0, torch.zeros_like(k0))
    k0_requant = mxint_quantizer(k0, block_size=16, element_bits=4, block_dim=-1)
    assert torch.allclose(k0, k0_requant, atol=1e-6)


def test_linear_partial_weight_config_raises():
    """A bucket with some-but-not-all weight keys is a typo, not a choice."""
    with pytest.raises(ValueError, match="incomplete weight config"):
        LinearMXInt(
            8,
            4,
            bias=False,
            config={
                "prefill": {"bypass": True},
                "decode": {"weight_block_size": 8},  # weight_width missing
            },
        )


def test_llama_force_decode_phase_scores_decode_numerics():
    """Full-sequence eval under force_runtime_phase('decode') must differ
    from the FP prefill result — this is the decode-chip accuracy proxy used
    by rotation search and DSE."""
    ref = _tiny_llama()
    model = copy.deepcopy(ref)
    model, _ = quantize_module_transform_pass(model, _llama_decode_only_pass_args())
    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        out_prefill = model(ids, use_cache=False).logits
        with force_runtime_phase("decode"):
            out_decode = model(ids, use_cache=False).logits
    assert not torch.allclose(out_prefill, out_decode)


def test_qwen3_decode_only_end_to_end_sdpa():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        head_dim=16,
        attention_dropout=0.0,
    )
    # KV + weights only quantised -> HF sdpa backend stays usable in BOTH phases.
    cfg._attn_implementation = "sdpa"
    ref = Qwen3ForCausalLM(cfg).eval()
    model = copy.deepcopy(ref)

    pass_args = {
        "by": "regex_name",
        r"model\.layers\.\d+\.self_attn$": {
            "config": {
                "name": "mxint",
                "prefill": {"bypass": True},
                "decode": {
                    "qk_matmul": {"bypass": True},
                    "av_matmul": {"bypass": True},
                    "rope": {"bypass": True},
                    "softmax": {"bypass": True},
                    "kv_cache": {"data_in_block_size": 16, "data_in_width": 4},
                },
            }
        },
        r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        r"|mlp\.(gate_proj|up_proj|down_proj))$": {
            "config": {
                "name": "mxint",
                "prefill": {"bypass": True},
                "decode": {
                    "weight_block_size": 16,
                    "weight_width": 4,
                    "data_in_block_size": 16,
                    "data_in_width": 8,
                },
            }
        },
        r"model\.layers\.\d+\.mlp$": {
            "config": {
                "name": "mxint",
                "prefill": {"bypass": True},
                "decode": {"data_in_exponent_width": 4, "data_in_frac_width": 3},
            }
        },
    }
    model, _ = quantize_module_transform_pass(model, pass_args)

    from chop.nn.quantized.modules.qwen3.mlp import Qwen3MLPMXInt

    assert isinstance(model.model.layers[0].mlp, Qwen3MLPMXInt)
    n_hooks = sum(
        1 for m in model.modules() if getattr(m, "_mase_phase_hook_installed", False)
    )
    assert n_hooks == 2

    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        out_q = model(ids, use_cache=False).logits
        out_ref = ref(ids, use_cache=False).logits
    assert torch.allclose(out_q, out_ref, atol=1e-6)

    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, do_sample=False)

    lin0 = model.model.layers[0].self_attn.q_proj
    assert lin0._decode_weight_q.numel() > 0
    assert torch.equal(lin0.weight, ref.model.layers[0].self_attn.q_proj.weight)


# ---------------------------------------------------------------------------
# DSE sweep format coverage
# ---------------------------------------------------------------------------

MXINT_SWEEP_WIDTHS = (2, 4, 8)
MXFP_SWEEP_FORMATS = {
    "E1M2": (1, 2),
    "E2M1": (2, 1),
    "E3M4": (3, 4),
    "E4M3": (4, 3),
    "E5M2": (5, 2),
}
FP_SETTING_FORMATS = {
    "E3M2": (3, 2),
    "E2M3": (2, 3),
    "E6M5": (6, 5),
    "E5M6": (5, 6),
    "E4M7": (4, 7),
    "E8M5": (8, 5),
}


@pytest.mark.parametrize("width", MXINT_SWEEP_WIDTHS)
def test_mxint_sweep_width_supported(width):
    x = torch.randn(4, 64) * 3
    q = mxint_quantizer(x, block_size=32, element_bits=width, block_dim=-1)
    assert torch.isfinite(q).all() and not torch.equal(q, x)
    assert len(torch.unique(q[0, :32])) <= 2**width


@pytest.mark.parametrize("fmt", sorted(MXFP_SWEEP_FORMATS))
def test_mxfp_sweep_format_supported(fmt):
    """Every MXFP element format of the W/A/KV sweep must run end to end
    through the decode weight-bank build and the activation path."""
    from chop.nn.quantizers import mxfp_quantizer

    exp_bits, frac_bits = MXFP_SWEEP_FORMATS[fmt]
    x = torch.randn(4, 64) * 3
    q = mxfp_quantizer(
        x,
        block_size=32,
        element_exp_bits=exp_bits,
        element_frac_bits=frac_bits,
        block_dim=-1,
    )
    assert torch.isfinite(q).all() and not torch.equal(q, x)

    src = nn.Linear(64, 32, bias=False)
    lin = LinearMXFP(
        64,
        32,
        bias=False,
        config={
            "prefill": {"bypass": True},
            "decode": {
                "weight_block_size": 32,
                "weight_exponent_width": exp_bits,
                "weight_frac_width": frac_bits,
                "data_in_block_size": 32,
                "data_in_exponent_width": exp_bits,
                "data_in_frac_width": frac_bits,
            },
        },
    )
    lin.load_state_dict(src.state_dict(), strict=False)
    assert lin._decode_weight_q.numel() > 0
    xin = torch.randn(2, 64)
    set_runtime_phase("prefill")
    assert torch.allclose(lin(xin), nn.functional.linear(xin, src.weight), atol=1e-6)
    set_runtime_phase("decode")
    assert torch.isfinite(lin(xin)).all()


@pytest.mark.parametrize("fmt", sorted(FP_SETTING_FORMATS))
def test_fp_setting_format_supported(fmt):
    """FP_SETTING = the vector-unit minifloat precision for non-linear ops
    and attention intermediates. It maps onto the softmax / rope stage
    configs, the MLP SiLU config, and the RMSNorm minifloat config."""
    from chop.nn.quantized.functional.rope import rope_minifloat
    from chop.nn.quantized.functional.silu import silu_minifloat
    from chop.nn.quantized.functional.softmax import softmax_minifloat

    exp_bits, frac_bits = FP_SETTING_FORMATS[fmt]
    cfg = {"data_in_exponent_width": exp_bits, "data_in_frac_width": frac_bits}

    attn = torch.randn(1, 2, 8, 8)
    s = softmax_minifloat(attn, cfg, dim=-1)
    assert torch.isfinite(s).all()
    g = silu_minifloat(torch.randn(4, 64), cfg)
    assert torch.isfinite(g).all()
    q, k = torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16)
    cos, sin = torch.ones(1, 8, 16), torch.zeros(1, 8, 16)
    rq, rk = rope_minifloat(q, k, cos, sin, cfg)
    assert torch.isfinite(rq).all() and torch.isfinite(rk).all()


def test_fp_setting_end_to_end_decode_only():
    """FP_SETTING applied through the module configs: softmax + rope in
    attention, SiLU in the MLP, and the RMSNorms, decode phase only."""
    ref = _tiny_llama()
    model = copy.deepcopy(ref)
    fp_setting = {"data_in_exponent_width": 6, "data_in_frac_width": 5}  # FP_E6M5
    pass_args = {
        "by": "regex_name",
        r"model\.layers\.\d+\.self_attn$": {
            "config": {
                "name": "mxint",
                "prefill": {"bypass": True},
                "decode": {
                    "qk_matmul": {"data_in_block_size": 16, "data_in_width": 8},
                    "av_matmul": {"data_in_block_size": 16, "data_in_width": 8},
                    "rope": dict(fp_setting),
                    "softmax": dict(fp_setting),
                    "kv_cache": {"data_in_block_size": 16, "data_in_width": 4},
                },
            }
        },
        r"model\.layers\.\d+\.mlp$": {
            "config": {
                "name": "mxint",
                "prefill": {"bypass": True},
                "decode": dict(fp_setting),
            }
        },
        r"model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)$": {
            "config": {
                "name": "minifloat",
                "prefill": {"bypass": True},
                "decode": {
                    "weight_exponent_width": 6,
                    "weight_frac_width": 5,
                    "data_in_exponent_width": 6,
                    "data_in_frac_width": 5,
                },
            }
        },
    }
    model, _ = quantize_module_transform_pass(model, pass_args)

    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        out_q = model(ids, use_cache=False).logits
        out_ref = ref(ids, use_cache=False).logits
    assert torch.equal(out_q, out_ref), "prefill stays exactly FP"

    with torch.no_grad():
        with force_runtime_phase("decode"):
            out_decode = model(ids, use_cache=False).logits
    assert torch.isfinite(out_decode).all()
    assert not torch.allclose(out_decode, out_ref), "FP_SETTING active in decode"

    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, do_sample=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
