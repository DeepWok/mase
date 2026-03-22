"""
Tests for the FlexAttention transform pass.

- Score-mod unit tests: CPU (pure logic, no compilation).
- Module replacement tests: CPU (verify pass swaps modules and preserves weights).
- Forward pass tests: CUDA only (torch.compile(flex_attention) requires GPU).
"""

import sys
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure the repo root is importable
sys.path.append(Path(__file__).resolve().parents[5].as_posix())

from chop.passes.module.transforms.attention.score_mods import (
    noop_score_mod,
    causal_score_mod,
    generate_sliding_window_score_mod,
    generate_alibi_score_mod,
    get_score_mod,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlexAttention requires CUDA"
)


# ===================================================================
# Score-mod unit tests (CPU)
# ===================================================================


class TestScoreMods:
    """Verify that each score_mod produces correct masking / bias patterns."""

    def test_noop_passes_through(self):
        score = torch.tensor(1.5)
        assert noop_score_mod(score, 0, 0, 3, 5) == score

    def test_causal_allows_past(self):
        score = torch.tensor(1.0)
        # q_idx >= kv_idx  →  score unchanged
        assert causal_score_mod(score, 0, 0, 5, 3) == score
        # q_idx == kv_idx  →  score unchanged
        assert causal_score_mod(score, 0, 0, 3, 3) == score

    def test_causal_blocks_future(self):
        score = torch.tensor(1.0)
        result = causal_score_mod(score, 0, 0, 2, 5)
        assert result == float("-inf")

    def test_sliding_window_within_window(self):
        mod = generate_sliding_window_score_mod(window_size=4)
        score = torch.tensor(2.0)
        # q=5, kv=3 → distance=2 < 4, and causal → pass
        assert mod(score, 0, 0, 5, 3) == score

    def test_sliding_window_outside_window(self):
        mod = generate_sliding_window_score_mod(window_size=4)
        score = torch.tensor(2.0)
        # q=10, kv=2 → distance=8 >= 4 → blocked
        result = mod(score, 0, 0, 10, 2)
        assert result == float("-inf")

    def test_sliding_window_blocks_future(self):
        mod = generate_sliding_window_score_mod(window_size=4)
        score = torch.tensor(2.0)
        result = mod(score, 0, 0, 2, 5)
        assert result == float("-inf")

    def test_alibi_adds_bias(self):
        num_heads = 8
        mod = generate_alibi_score_mod(num_heads)
        score = torch.tensor(0.0)
        # Head 0, q=5, kv=3 → distance = 2, slope > 0 → positive bias
        result = mod(score, 0, 0, 5, 3)
        assert result > 0.0
        # Head 0, q=3, kv=5 → distance = -2 → negative bias
        result_neg = mod(score, 0, 0, 3, 5)
        assert result_neg < 0.0

    def test_get_score_mod_registry(self):
        assert get_score_mod("none") is noop_score_mod
        assert get_score_mod("causal") is causal_score_mod

    def test_get_score_mod_factory(self):
        mod = get_score_mod("sliding_window", window_size=8)
        assert callable(mod)

    def test_get_score_mod_invalid(self):
        with pytest.raises(ValueError, match="Unknown score_mod"):
            get_score_mod("nonexistent")


# ===================================================================
# Module replacement tests (CPU — no forward pass, no torch.compile)
# ===================================================================


class TestLlamaModuleReplacement:
    """Test that the pass correctly replaces Llama attention modules on CPU."""

    def _make_tiny_llama(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            vocab_size=256,
            _attn_implementation="eager",
        )
        return LlamaForCausalLM(config).to(torch.float32)

    def test_pass_replaces_modules(self):
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
            _build_llama_flex_attention_cls,
        )

        model = self._make_tiny_llama()
        LlamaFlexAttention, _, _ = _build_llama_flex_attention_cls()

        pass_args = {"score_mod": "causal"}
        model, stats = flex_attention_transform_pass(model, pass_args)

        assert stats["llama_replaced"] == 2  # 2 layers

        # Verify all attention modules are now LlamaFlexAttention
        for name, module in model.named_modules():
            if "self_attn" in name and not any(
                sub in name for sub in ["q_proj", "k_proj", "v_proj", "o_proj", "rotary"]
            ):
                if hasattr(module, "score_mod_fn"):
                    assert type(module).__name__ == "LlamaFlexAttention"

    def test_weights_preserved_after_transform(self):
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        model = self._make_tiny_llama()

        # Capture original weights
        original_weights = {}
        for name, module in model.named_modules():
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                original_weights[name] = {
                    "q": module.q_proj.weight.clone(),
                    "k": module.k_proj.weight.clone(),
                    "v": module.v_proj.weight.clone(),
                    "o": module.o_proj.weight.clone(),
                }

        pass_args = {"score_mod": "causal"}
        model, _ = flex_attention_transform_pass(model, pass_args)

        # Verify weights are identical after transform
        for name, module in model.named_modules():
            if name in original_weights and hasattr(module, "q_proj"):
                assert torch.equal(module.q_proj.weight, original_weights[name]["q"])
                assert torch.equal(module.k_proj.weight, original_weights[name]["k"])
                assert torch.equal(module.v_proj.weight, original_weights[name]["v"])
                assert torch.equal(module.o_proj.weight, original_weights[name]["o"])

    def test_score_mod_attached(self):
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        model = self._make_tiny_llama()
        pass_args = {"score_mod": "causal"}
        model, _ = flex_attention_transform_pass(model, pass_args)

        for name, module in model.named_modules():
            if hasattr(module, "score_mod_fn") and module.score_mod_fn is not None:
                assert module.score_mod_fn is causal_score_mod


class TestMistralModuleReplacement:
    """Test that the pass correctly replaces Mistral attention modules on CPU."""

    def _make_tiny_mistral(self):
        from transformers import MistralConfig, MistralForCausalLM

        config = MistralConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            vocab_size=256,
            sliding_window=32,
            _attn_implementation="eager",
        )
        return MistralForCausalLM(config).to(torch.float32)

    def test_pass_replaces_modules(self):
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
            _build_mistral_flex_attention_cls,
        )

        model = self._make_tiny_mistral()
        MistralFlexAttention, _, _ = _build_mistral_flex_attention_cls()

        pass_args = {
            "score_mod": "sliding_window",
            "score_mod_kwargs": {"window_size": 32},
        }
        model, stats = flex_attention_transform_pass(model, pass_args)

        assert stats["mistral_replaced"] == 2

    def test_weights_preserved_after_transform(self):
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        model = self._make_tiny_mistral()

        original_weights = {}
        for name, module in model.named_modules():
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                original_weights[name] = {
                    "q": module.q_proj.weight.clone(),
                    "k": module.k_proj.weight.clone(),
                    "v": module.v_proj.weight.clone(),
                    "o": module.o_proj.weight.clone(),
                }

        pass_args = {
            "score_mod": "sliding_window",
            "score_mod_kwargs": {"window_size": 32},
        }
        model, _ = flex_attention_transform_pass(model, pass_args)

        for name, module in model.named_modules():
            if name in original_weights and hasattr(module, "q_proj"):
                assert torch.equal(module.q_proj.weight, original_weights[name]["q"])
                assert torch.equal(module.k_proj.weight, original_weights[name]["k"])
                assert torch.equal(module.v_proj.weight, original_weights[name]["v"])
                assert torch.equal(module.o_proj.weight, original_weights[name]["o"])


class TestBertModuleReplacement:
    """Test that the pass correctly replaces BERT attention modules on CPU."""

    def _make_tiny_bert(self):
        from chop.models.bert.modeling_bert import BertModel
        from chop.models.bert.configuration_bert import BertConfig

        config = BertConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            vocab_size=256,
        )
        return BertModel(config).to(torch.float32)

    def test_pass_replaces_modules(self):
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        model = self._make_tiny_bert()
        pass_args = {"score_mod": "none"}
        model, stats = flex_attention_transform_pass(model, pass_args)

        assert stats["bert_replaced"] == 2

    def test_weights_preserved_after_transform(self):
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        model = self._make_tiny_bert()

        original_weights = {}
        for name, module in model.named_modules():
            if hasattr(module, "query") and hasattr(module, "key"):
                original_weights[name] = {
                    "query": module.query.weight.clone(),
                    "key": module.key.weight.clone(),
                    "value": module.value.weight.clone(),
                }

        pass_args = {"score_mod": "none"}
        model, _ = flex_attention_transform_pass(model, pass_args)

        for name, module in model.named_modules():
            if name in original_weights and hasattr(module, "query"):
                assert torch.equal(module.query.weight, original_weights[name]["query"])
                assert torch.equal(module.key.weight, original_weights[name]["key"])
                assert torch.equal(module.value.weight, original_weights[name]["value"])


# ===================================================================
# Forward pass tests — require CUDA
# ===================================================================


@requires_cuda
class TestLlamaFlexForward:
    """Test FlexAttention forward pass on Llama (CUDA only)."""

    def test_forward_produces_correct_shape(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            vocab_size=256,
            _attn_implementation="eager",
        )
        model = LlamaForCausalLM(config).to("cuda").to(torch.float32)
        pass_args = {"score_mod": "causal"}
        model, _ = flex_attention_transform_pass(model, pass_args)

        input_ids = torch.randint(0, 256, (2, 16), device="cuda")
        with torch.no_grad():
            output = model(input_ids)

        assert output.logits.shape == (2, 16, 256)


@requires_cuda
class TestMistralFlexForward:
    """Test FlexAttention forward pass on Mistral (CUDA only)."""

    def test_forward_produces_correct_shape(self):
        from transformers import MistralConfig, MistralForCausalLM
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        config = MistralConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            vocab_size=256,
            sliding_window=32,
            _attn_implementation="eager",
        )
        model = MistralForCausalLM(config).to("cuda").to(torch.float32)
        pass_args = {
            "score_mod": "sliding_window",
            "score_mod_kwargs": {"window_size": 32},
        }
        model, _ = flex_attention_transform_pass(model, pass_args)

        input_ids = torch.randint(0, 256, (2, 16), device="cuda")
        with torch.no_grad():
            output = model(input_ids)

        assert output.logits.shape == (2, 16, 256)


@requires_cuda
class TestBertFlexForward:
    """Test FlexAttention forward pass on BERT (CUDA only)."""

    def test_forward_produces_correct_shape(self):
        from chop.models.bert.modeling_bert import BertModel
        from chop.models.bert.configuration_bert import BertConfig
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        config = BertConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            vocab_size=256,
        )
        model = BertModel(config).to("cuda").to(torch.float32)
        pass_args = {"score_mod": "none"}
        model, _ = flex_attention_transform_pass(model, pass_args)

        input_ids = torch.randint(0, 256, (2, 16), device="cuda")
        with torch.no_grad():
            output = model(input_ids)

        assert output.last_hidden_state.shape == (2, 16, 64)
