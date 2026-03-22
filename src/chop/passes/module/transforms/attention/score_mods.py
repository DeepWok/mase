"""
Score modification functions for PyTorch FlexAttention.

Each score_mod follows the flex_attention signature:
    score_mod(score, b, h, q_idx, kv_idx) -> modified_score

where:
    score  : scalar attention score (pre-softmax)
    b      : batch index
    h      : head index
    q_idx  : query position index
    kv_idx : key/value position index

These are pure functions suitable for torch.compile tracing.
"""

import torch
import math


# ---------------------------------------------------------------------------
# Score mod implementations
# ---------------------------------------------------------------------------


def noop_score_mod(score, b, h, q_idx, kv_idx):
    """Identity — no masking, no bias. Full bidirectional attention."""
    return score


def causal_score_mod(score, b, h, q_idx, kv_idx):
    """Standard causal (autoregressive) mask: attend only to past + current."""
    return torch.where(
        torch.as_tensor(q_idx >= kv_idx), score, torch.tensor(float("-inf"))
    )


def generate_sliding_window_score_mod(window_size: int):
    """
    Factory for sliding-window causal attention.

    Tokens attend only to the previous `window_size` positions (inclusive).
    Positions outside the window receive -inf.

    Args:
        window_size: Number of past tokens each position can attend to.
    """

    def sliding_window_score_mod(score, b, h, q_idx, kv_idx):
        causal_mask = torch.as_tensor(q_idx >= kv_idx)
        window_mask = torch.as_tensor((q_idx - kv_idx) < window_size)
        return torch.where(causal_mask & window_mask, score, torch.tensor(float("-inf")))

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
    # Pre-compute slopes on meta device — they are constants.
    base = 2.0 ** (-(2.0 ** -(math.log2(num_heads) - 3)))
    slopes = torch.tensor(
        [base ** (i + 1) for i in range(num_heads)], dtype=torch.float32
    )

    def alibi_score_mod(score, b, h, q_idx, kv_idx):
        bias = (q_idx - kv_idx) * slopes[h]
        return score + bias

    return alibi_score_mod


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
