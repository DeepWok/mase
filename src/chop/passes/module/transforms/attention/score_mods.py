"""
Score modification functions for PyTorch FlexAttention.

Each score_mod follows the flex_attention signature:
    score_mod(score, b, h, q_idx, kv_idx) -> modified_score

Each mask_mod follows the create_block_mask signature:
    mask_mod(b, h, q_idx, kv_idx) -> bool

where:
    score  : scalar attention score (pre-softmax)
    b      : batch index
    h      : head index
    q_idx  : query position index
    kv_idx : key/value position index

score_mods handle fine-grained score modification within computed blocks.
mask_mods define which blocks to skip entirely (the source of speedup).

These are pure functions suitable for torch.compile tracing.
"""

import torch
import math


# ---------------------------------------------------------------------------
# Score mod implementations
# ---------------------------------------------------------------------------


def noop_score_mod(score, b, h, q_idx, kv_idx):
    """Identity -- no masking, no bias. Full bidirectional attention."""
    return score


# def causal_score_mod(score, b, h, q_idx, kv_idx):
#     """Standard causal (autoregressive) mask: attend only to past + current."""
#     return torch.where(torch.as_tensor(q_idx >= kv_idx), score, -float("inf"))

def causal_score_mod(score, b, h, q_idx, kv_idx):
    """Standard causal mask: attend only to past + current."""
    return torch.where(q_idx >= kv_idx, score, float("-inf"))


def generate_sliding_window_score_mod(window_size: int):
    """
    Factory for sliding-window causal attention.

    Tokens attend only to the previous `window_size` positions (inclusive).
    Positions outside the window receive -inf.

    Args:
        window_size: Number of past tokens each position can attend to.
    """

    # def sliding_window_score_mod(score, b, h, q_idx, kv_idx):
    #     causal_mask = q_idx >= kv_idx
    #     window_mask = (q_idx - kv_idx) < window_size
    #     return torch.where(torch.as_tensor(causal_mask & window_mask), score, -float("inf"))
    
    def sliding_window_score_mod(score, b, h, q_idx, kv_idx):
        # A single inline return expression. 
        # No variable reassignments, no torch.as_tensor(). 
        # This compiles cleanly to Triton.
        return torch.where((q_idx >= kv_idx) & ((q_idx - kv_idx) < window_size), score, float("-inf"))

    return sliding_window_score_mod


def generate_alibi_score_mod(num_heads: int):
    """
    Factory for ALiBi (Attention with Linear Biases).

    Adds a head-specific linear bias proportional to the distance
    between query and key positions.  Slopes follow the geometric
    series from the ALiBi paper (Press et al., 2022).

    Args:
        num_heads: Total number of attention heads.
    """
    
    # 1. Compute the math once, outside the kernel
    base = 2.0 ** (-(2.0 ** -(math.log2(num_heads) - 3)))
    
    # 2. Materialize the slopes directly on the GPU
    slopes = torch.tensor(
        [base ** (i + 1) for i in range(num_heads)], 
        dtype=torch.float32, 
        device="cuda"
    )

    def alibi_score_mod(score, b, h, q_idx, kv_idx):
        # 3. A simple scalar multiplication.
        # Autograd easily tracks this with almost zero memory overhead.
        bias = (q_idx - kv_idx) * slopes[h]
        return score + bias

    return alibi_score_mod


# ---------------------------------------------------------------------------
# Mask mod implementations (for create_block_mask -- block-level sparsity)
# ---------------------------------------------------------------------------


def noop_mask_mod(b, h, q_idx, kv_idx):
    """Full attention -- all blocks computed."""
    return True


def causal_mask_mod(b, h, q_idx, kv_idx):
    """Causal mask: attend only to past + current positions."""
    return q_idx >= kv_idx


def generate_sliding_window_mask_mod(window_size: int):
    """
    Factory for sliding-window causal mask_mod.

    Returns a mask_mod that marks positions as True only if they are
    causal AND within the sliding window. Used with create_block_mask
    to skip entire blocks that fall outside the window.

    Args:
        window_size: Number of past tokens each position can attend to.
    """

    def sliding_window_mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) < window_size)

    return sliding_window_mask_mod

def generate_document_mask_mod(doc_len: int):
    """
    Mask mod for Document Masking. Tells the GPU to physically skip 
    computing blocks that cross document boundaries.
    """
    def document_mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx // doc_len) == (kv_idx // doc_len))

    return document_mask_mod


def generate_alibi_sliding_window_score_mod(num_heads: int, window_size: int):
    """Pure math compound ALiBi + SWA."""
    base = 2.0 ** (-(2.0 ** -(math.log2(num_heads) - 3)))
    
    # Force device="cuda" so TorchInductor doesn't crash fetching CPU memory
    slopes = torch.tensor(
        [base ** (i + 1) for i in range(num_heads)], dtype=torch.float32, device="cuda"
    )

    def alibi_sliding_window_score_mod(score, b, h, q_idx, kv_idx):
        # A simple scalar multiplication. Zero gradient overhead.
        bias = (q_idx - kv_idx) * slopes[h]
        return torch.where((q_idx >= kv_idx) & ((q_idx - kv_idx) < window_size), score + bias, float("-inf"))

    return alibi_sliding_window_score_mod

def generate_document_mask_score_mod(doc_len: int):
    """
    Factory for Document Masking (Sequence Packing).
    Tokens can only attend to past tokens within their specific document chunk.
    """
    def document_score_mod(score, b, h, q_idx, kv_idx):
        # Tokens are in the same document if their index divided by doc_len is equal
        same_doc = (q_idx // doc_len) == (kv_idx // doc_len)
        causal = q_idx >= kv_idx
        return torch.where(same_doc & causal, score, float("-inf"))

    return document_score_mod



# ---------------------------------------------------------------------------
# Registry & accessor
# ---------------------------------------------------------------------------

# Direct (non-parameterised) score mods
_SCORE_MOD_REGISTRY = {
    "none": noop_score_mod,
    "causal": causal_score_mod,
}

# Factory functions for parameterised score mods
_SCORE_MOD_FACTORY = {
    "sliding_window": generate_sliding_window_score_mod,
    "alibi": generate_alibi_score_mod,
    "alibi_sliding_window": generate_alibi_sliding_window_score_mod,
    "document_mask": generate_document_mask_score_mod,
}


# Direct (non-parameterised) mask mods
_MASK_MOD_REGISTRY = {
    "none": noop_mask_mod,
    "causal": causal_mask_mod,
}

# Factory functions for parameterised mask mods
_MASK_MOD_FACTORY = {
    "sliding_window": generate_sliding_window_mask_mod,
    "document_mask": generate_document_mask_mod,
}


def get_score_mod(name: str, **kwargs):
    """
    Retrieve a score_mod function by name.

    For parameterised score mods (sliding_window, alibi), pass the required
    kwargs (e.g. ``window_size=256`` or ``num_heads=32``).

    Args:
        name: One of "none", "causal", "sliding_window", "alibi".
        **kwargs: Forwarded to the factory for parameterised variants.

    Returns:
        A callable with signature ``(score, b, h, q_idx, kv_idx) -> score``.

    Raises:
        ValueError: If *name* is not recognised.
    """
    if name in _SCORE_MOD_REGISTRY:
        return _SCORE_MOD_REGISTRY[name]
    if name in _SCORE_MOD_FACTORY:
        return _SCORE_MOD_FACTORY[name](**kwargs)
    valid = sorted(set(_SCORE_MOD_REGISTRY) | set(_SCORE_MOD_FACTORY))
    raise ValueError(f"Unknown score_mod '{name}'. Choose from {valid}.")


def get_mask_mod(name: str, **kwargs):
    """
    Retrieve a mask_mod function by name.

    mask_mods are used with ``create_block_mask()`` to define block-level
    sparsity patterns that tell FlexAttention which blocks to skip entirely.

    Args:
        name: One of "none", "causal", "sliding_window".
        **kwargs: Forwarded to the factory for parameterised variants
                  (e.g. ``window_size=256``).

    Returns:
        A callable with signature ``(b, h, q_idx, kv_idx) -> bool``.

    Raises:
        ValueError: If *name* is not recognised.
    """
    if name in _MASK_MOD_REGISTRY:
        return _MASK_MOD_REGISTRY[name]
    if name in _MASK_MOD_FACTORY:
        return _MASK_MOD_FACTORY[name](**kwargs)
    valid = sorted(set(_MASK_MOD_REGISTRY) | set(_MASK_MOD_FACTORY))
    raise ValueError(f"Unknown mask_mod '{name}'. Choose from {valid}.")
