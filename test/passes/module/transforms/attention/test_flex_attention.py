"""
Tests for the FlexAttention transform pass.

- Score-mod unit tests: CPU (pure logic, no compilation).
- Mask-mod unit tests: CPU (pure logic for block_mask patterns).
- Module replacement tests: CPU (verify pass swaps modules and preserves weights).
- Forward pass tests: CUDA only (torch.compile(flex_attention) requires GPU).
- Training step tests: CUDA only (backward pass through compiled flex_attention).
- Longer sequence tests: CUDA only (seq_len=512).
- BFloat16 tests: CUDA only (bf16 inference and training).
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
    noop_mask_mod,
    causal_mask_mod,
    generate_sliding_window_mask_mod,
    get_mask_mod,
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
        q_idx, kv_idx = torch.tensor(3), torch.tensor(5)
        assert noop_score_mod(score, 0, 0, q_idx, kv_idx) == score

    def test_causal_allows_past(self):
        score = torch.tensor(1.0)
        # q_idx >= kv_idx  ->  score unchanged
        q_idx, kv_idx = torch.tensor(5), torch.tensor(3)
        assert causal_score_mod(score, 0, 0, q_idx, kv_idx) == score
        
        # q_idx == kv_idx  ->  score unchanged
        q_idx, kv_idx = torch.tensor(3), torch.tensor(3)
        assert causal_score_mod(score, 0, 0, q_idx, kv_idx) == score

    def test_causal_blocks_future(self):
        score = torch.tensor(1.0)
        q_idx, kv_idx = torch.tensor(2), torch.tensor(5)
        result = causal_score_mod(score, 0, 0, q_idx, kv_idx)
        assert result == float("-inf")

    def test_sliding_window_within_window(self):
        mod = generate_sliding_window_score_mod(window_size=4)
        score = torch.tensor(2.0)
        # q=5, kv=3 -> distance=2 < 4, and causal -> pass
        q_idx, kv_idx = torch.tensor(5), torch.tensor(3)
        assert mod(score, 0, 0, q_idx, kv_idx) == score

    def test_sliding_window_outside_window(self):
        mod = generate_sliding_window_score_mod(window_size=4)
        score = torch.tensor(2.0)
        # q=10, kv=2 -> distance=8 >= 4 -> blocked
        q_idx, kv_idx = torch.tensor(10), torch.tensor(2)
        result = mod(score, 0, 0, q_idx, kv_idx)
        assert result == float("-inf")

    def test_sliding_window_blocks_future(self):
        mod = generate_sliding_window_score_mod(window_size=4)
        score = torch.tensor(2.0)
        q_idx, kv_idx = torch.tensor(2), torch.tensor(5)
        result = mod(score, 0, 0, q_idx, kv_idx)
        assert result == float("-inf")

    def test_alibi_adds_bias(self):
        num_heads = 8
        mod = generate_alibi_score_mod(num_heads)
        score = torch.tensor(0.0)
        
        # Head 0, q=5, kv=3 -> distance = 2, slope > 0 -> positive bias
        q_idx, kv_idx = torch.tensor(5), torch.tensor(3)
        result = mod(score, 0, 0, q_idx, kv_idx)
        assert result > 0.0
        
        # Head 0, q=3, kv=5 -> distance = -2 -> negative bias
        q_idx, kv_idx = torch.tensor(3), torch.tensor(5)
        result_neg = mod(score, 0, 0, q_idx, kv_idx)
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
# Mask-mod unit tests (CPU)
# ===================================================================


class TestMaskMods:
    """Verify that each mask_mod produces correct boolean patterns."""

    def test_noop_always_true(self):
        assert noop_mask_mod(0, 0, 3, 5) is True
        assert noop_mask_mod(0, 0, 5, 3) is True

    def test_causal_allows_past(self):
        assert causal_mask_mod(0, 0, 5, 3) is True
        assert causal_mask_mod(0, 0, 3, 3) is True

    def test_causal_blocks_future(self):
        assert causal_mask_mod(0, 0, 2, 5) is False

    def test_sliding_window_within_window(self):
        mod = generate_sliding_window_mask_mod(window_size=4)
        # q=5, kv=3 -> distance=2 < 4, and causal -> True
        assert mod(0, 0, 5, 3) is True

    def test_sliding_window_outside_window(self):
        mod = generate_sliding_window_mask_mod(window_size=4)
        # q=10, kv=2 -> distance=8 >= 4 -> False
        assert mod(0, 0, 10, 2) is False

    def test_sliding_window_blocks_future(self):
        mod = generate_sliding_window_mask_mod(window_size=4)
        assert mod(0, 0, 2, 5) is False

    def test_get_mask_mod_registry(self):
        assert get_mask_mod("none") is noop_mask_mod
        assert get_mask_mod("causal") is causal_mask_mod

    def test_get_mask_mod_factory(self):
        mod = get_mask_mod("sliding_window", window_size=8)
        assert callable(mod)

    def test_get_mask_mod_invalid(self):
        with pytest.raises(ValueError, match="Unknown mask_mod"):
            get_mask_mod("nonexistent")


# ===================================================================
# Module replacement tests (CPU -- no forward pass, no torch.compile)
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

    def test_mask_mod_attached(self):
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        model = self._make_tiny_llama()
        pass_args = {"score_mod": "causal"}
        model, _ = flex_attention_transform_pass(model, pass_args)

        for name, module in model.named_modules():
            if hasattr(module, "mask_mod_fn") and module.mask_mod_fn is not None:
                assert module.mask_mod_fn is causal_mask_mod

    def test_block_mask_disabled(self):
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        model = self._make_tiny_llama()
        pass_args = {"score_mod": "causal", "use_block_mask": False}
        model, _ = flex_attention_transform_pass(model, pass_args)

        for name, module in model.named_modules():
            if hasattr(module, "mask_mod_fn"):
                assert module.mask_mod_fn is None


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
# Forward pass tests -- require CUDA
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

        # MASE BertModel uses a passthrough embedding (no actual embedding
        # layer), so we feed float hidden states directly, not integer IDs.
        hidden_states = torch.randn(2, 16, 64, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            output = model(hidden_states)

        assert output.shape == (2, 16, 64)


# ===================================================================
# Extended CUDA validation tests
# ===================================================================


@requires_cuda
class TestFlexTrainingStep:
    """Verify that gradients flow through compiled flex_attention (1 training step)."""

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
        return LlamaForCausalLM(config).to("cuda").to(torch.float32)

    def test_llama_backward_pass(self):
        """One forward + backward step to confirm gradients propagate."""
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )

        model = self._make_tiny_llama()
        pass_args = {"score_mod": "causal"}
        model, _ = flex_attention_transform_pass(model, pass_args)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 256, (2, 16), device="cuda")
        labels = input_ids.clone()

        # Forward
        output = model(input_ids, labels=labels)
        loss = output.loss
        assert loss is not None and loss.requires_grad

        # Backward
        loss.backward()

        # Verify gradients exist on attention projection weights
        for name, param in model.named_parameters():
            if "q_proj" in name and param.grad is not None:
                assert param.grad.abs().sum() > 0, f"Zero gradient on {name}"
                break

        # Optimizer step (no crash = success)
        optimizer.step()
        optimizer.zero_grad()

    def test_mistral_backward_pass(self):
        """One forward + backward step for Mistral with sliding_window score_mod."""
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
        model.train()

        input_ids = torch.randint(0, 256, (2, 16), device="cuda")
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        loss = output.loss
        assert loss is not None and loss.requires_grad
        loss.backward()

        for name, param in model.named_parameters():
            if "q_proj" in name and param.grad is not None:
                assert param.grad.abs().sum() > 0, f"Zero gradient on {name}"
                break


@requires_cuda
class TestFlexLongerSequence:
    """Test with longer sequence lengths (closer to real usage)."""

    def test_llama_seq512(self):
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
            max_position_embeddings=512,
            vocab_size=256,
            _attn_implementation="eager",
        )
        model = LlamaForCausalLM(config).to("cuda").to(torch.float32)
        pass_args = {"score_mod": "causal"}
        model, _ = flex_attention_transform_pass(model, pass_args)

        input_ids = torch.randint(0, 256, (1, 512), device="cuda")
        with torch.no_grad():
            output = model(input_ids)

        assert output.logits.shape == (1, 512, 256)


@requires_cuda
class TestFlexBFloat16:
    """Test with bf16 precision (common for training on modern GPUs)."""

    def test_llama_bf16_forward(self):
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
        model = LlamaForCausalLM(config).to("cuda").to(torch.bfloat16)
        pass_args = {"score_mod": "causal"}
        model, _ = flex_attention_transform_pass(model, pass_args)

        input_ids = torch.randint(0, 256, (2, 16), device="cuda")
        with torch.no_grad():
            output = model(input_ids)

        assert output.logits.shape == (2, 16, 256)
        assert output.logits.dtype == torch.bfloat16

    def test_llama_bf16_training_step(self):
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
        model = LlamaForCausalLM(config).to("cuda").to(torch.bfloat16)
        pass_args = {"score_mod": "causal"}
        model, _ = flex_attention_transform_pass(model, pass_args)
        model.train()

        input_ids = torch.randint(0, 256, (2, 16), device="cuda")
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        loss = output.loss
        loss.backward()

        # Confirm gradients exist and are bf16
        for name, param in model.named_parameters():
            if "q_proj" in name and param.grad is not None:
                assert param.grad.dtype == torch.bfloat16
                assert param.grad.abs().sum() > 0
                break


# ===================================================================
# Block mask CUDA tests (seq_len >= 128 to exercise create_block_mask)
# ===================================================================


@requires_cuda
class TestBlockMaskCUDA:
    """Validate that block_mask (create_block_mask) works on CUDA with seq_len >= 128."""

    def test_llama_causal_block_mask(self):
        """Llama forward with causal block_mask at seq_len=128."""
        from transformers import LlamaConfig, LlamaForCausalLM
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
            _compiled_flex_attention,
        )
        import chop.passes.module.transforms.attention.flex_attention_transform as fat

        fat._compiled_flex_attention = None  # reset compile cache

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            vocab_size=256,
            _attn_implementation="eager",
        )
        model = LlamaForCausalLM(config).to("cuda").to(torch.float32)
        pass_args = {"score_mod": "causal"}  # block_mask enabled by default
        model, _ = flex_attention_transform_pass(model, pass_args)

        input_ids = torch.randint(0, 256, (2, 128), device="cuda")
        with torch.no_grad():
            output = model(input_ids)

        assert output.logits.shape == (2, 128, 256)

    def test_llama_sliding_window_block_mask(self):
        """Llama forward with sliding_window block_mask at seq_len=128."""
        from transformers import LlamaConfig, LlamaForCausalLM
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )
        import chop.passes.module.transforms.attention.flex_attention_transform as fat

        fat._compiled_flex_attention = None

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            vocab_size=256,
            _attn_implementation="eager",
        )
        model = LlamaForCausalLM(config).to("cuda").to(torch.float32)
        pass_args = {
            "score_mod": "sliding_window",
            "score_mod_kwargs": {"window_size": 64},
        }
        model, _ = flex_attention_transform_pass(model, pass_args)

        input_ids = torch.randint(0, 256, (2, 128), device="cuda")
        with torch.no_grad():
            output = model(input_ids)

        assert output.logits.shape == (2, 128, 256)

    def test_block_mask_training_step(self):
        """Training step with block_mask to confirm gradients flow."""
        from transformers import LlamaConfig, LlamaForCausalLM
        from chop.passes.module.transforms.attention.flex_attention_transform import (
            flex_attention_transform_pass,
        )
        import chop.passes.module.transforms.attention.flex_attention_transform as fat

        fat._compiled_flex_attention = None

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            vocab_size=256,
            _attn_implementation="eager",
        )
        model = LlamaForCausalLM(config).to("cuda").to(torch.float32)
        pass_args = {"score_mod": "causal"}
        model, _ = flex_attention_transform_pass(model, pass_args)
        model.train()

        input_ids = torch.randint(0, 256, (2, 128), device="cuda")
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        loss = output.loss
        assert loss is not None and loss.requires_grad
        loss.backward()

        for name, param in model.named_parameters():
            if "q_proj" in name and param.grad is not None:
                assert param.grad.abs().sum() > 0, f"Zero gradient on {name}"
                break
