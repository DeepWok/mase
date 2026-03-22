# FlexAttention Integration Plan for MASE

## Overview

Create a standalone `flex_attention_transform_pass` that walks a model, finds existing attention modules (Llama, Mistral, BERT), and replaces them with FlexAttention-based variants. The FlexAttention variants inherit from the base attention classes (weights stay untouched) and override only the forward method to use `torch.compile(flex_attention)` with configurable `score_mod` functions.

## Architecture

```
The pass does NOT change model weights or projections.
It only changes how Q, K, V are combined — the kernel dispatch.

LlamaAttention (base)               LlamaFlexAttention (new)
├── q_proj, k_proj, v_proj, o_proj   ├── same weights (inherited)
├── rotary_emb                       ├── same rotary_emb (inherited)
└── forward: manual matmul+softmax   └── forward: torch.compile(flex_attention) + score_mod
```

## Files to Create

### 1. `src/chop/passes/module/transforms/attention/score_mods.py` (~80 LOC)

Standalone `score_mod` functions compatible with `flex_attention` API:

```python
def causal_score_mod(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float("-inf"))

def sliding_window_score_mod(window_size):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            (q_idx - kv_idx >= 0) & (q_idx - kv_idx < window_size),
            score, float("-inf")
        )
    return score_mod

def alibi_score_mod(num_heads):
    def score_mod(score, b, h, q_idx, kv_idx):
        bias = (q_idx - kv_idx) * slopes[h]  # pre-computed slopes
        return score + bias
    return score_mod
```

Plus a registry:
```python
SCORE_MOD_REGISTRY = {"causal": ..., "sliding_window": ..., "alibi": ..., "none": ...}
def get_score_mod(name, **kwargs) -> callable
```

### 2. `src/chop/passes/module/transforms/attention/flex_attention_transform.py` (~200 LOC)

**Key design**: Each FlexAttention variant subclasses the base attention class from the model (same pattern as `LlamaSdpaAttention(LlamaAttention)`). This means:
- All weight projections (q/k/v/o_proj) are inherited — zero weight transfer needed
- Only the `forward()` method is overridden to use `flex_attention` instead of manual matmul or SDPA

**Pass function**:
```python
def flex_attention_transform_pass(network, pass_args):
    """
    pass_args = {
        "score_mod": "causal",           # or "sliding_window", "alibi", "none"
        "score_mod_kwargs": {"window_size": 256},  # kwargs for parameterized score_mods
    }
    """
```

Walk all modules in the network:
- `LlamaAttention` / `LlamaSdpaAttention` → replace with `LlamaFlexAttention`
- `MistralAttention` / `MistralSdpaAttention` → replace with `MistralFlexAttention`
- `BertSelfAttention` → replace with `BertFlexSelfAttention`

**Replacement strategy**: For each found module:
1. Instantiate the FlexAttention subclass with the same `config` and `layer_idx`
2. Copy all parameters via `new_module.load_state_dict(old_module.state_dict())`
3. Set the `score_mod` function on the new module
4. Replace in-place using `set_module_by_name`

**FlexAttention forward (Llama/Mistral)**:
```python
class LlamaFlexAttention(LlamaAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self._compiled_flex_attention = torch.compile(flex_attention)
        self.score_mod_fn = None  # set by the pass

    def forward(self, hidden_states, attention_mask=None, position_ids=None, ...):
        # Q, K, V projection + reshape + RoPE (same as base class)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache handling (same as base)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # GQA: repeat KV heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # === THIS IS THE ONLY CHANGE ===
        attn_output = self._compiled_flex_attention(
            query_states, key_states, value_states,
            score_mod=self.score_mod_fn
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)
```

**BERT variant**: Similar but no RoPE, no GQA, different projection names (`query`/`key`/`value` instead of `q_proj`/`k_proj`/`v_proj`).

### 3. `src/chop/passes/module/transforms/attention/__init__.py` — EDIT

Add export:
```python
from .flex_attention_transform import flex_attention_transform_pass
```

### 4. `src/chop/passes/module/transforms/__init__.py` — EDIT

Add export:
```python
from .attention import flex_attention_transform_pass
```

### 5. `test/passes/module/transforms/attention/test_flex_attention.py` (~150 LOC)

Tests:
- **Unit test score_mods**: Verify each score_mod function produces correct masking patterns on small tensors
- **Llama pass test**: Load TinyLlama config, create model, apply pass, verify all attention modules are `LlamaFlexAttention`, verify forward produces correct-shaped output
- **Mistral pass test**: Same for Mistral with sliding_window score_mod
- **BERT pass test**: Same for BERT with no causal masking
- **Numerical correctness test**: Compare flex_attention output vs eager attention on small inputs (within tolerance)

Tests that require GPU/CUDA marked with `@pytest.mark.skipif(not torch.cuda.is_available())` since `flex_attention` + `torch.compile` requires CUDA.

## Key Technical Details

1. **torch.compile scope**: We compile ONLY `flex_attention` at function level, NOT the whole model. This avoids graph break issues documented in MASE.

2. **flex_attention API** (torch 2.6+):
   ```python
   from torch.nn.attention.flex_attention import flex_attention, create_block_mask
   ```

3. **GQA handling**: `flex_attention` natively supports different numbers of Q and KV heads — we may not need `repeat_kv` when using flex_attention directly, which is a performance win.

4. **Weight transfer**: Since we subclass the base attention class and use `load_state_dict`, all weights transfer exactly with zero computation.

5. **Return signature**: Must match the original — Llama returns `attn_output` (single tensor after recent changes), Mistral returns `(attn_output, None, past_key_value)`, BERT returns `context_layer`.

## Implementation Order

1. `score_mods.py` — pure functions, no dependencies
2. `flex_attention_transform.py` — the FlexAttention classes + pass function
3. Update `__init__.py` files for exports
4. `test_flex_attention.py` — tests
