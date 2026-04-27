"""
Fused Residual Addition + RMSNorm Triton Kernel
================================================

Part 2 of the ADLS kernel-fusion-aware optimisation pipeline in MASE.

This kernel fuses two operations that occur sequentially in every transformer
decoder layer into a single GPU kernel launch:

    1. Residual addition:   residual = residual + hidden_states
    2. RMS normalisation:   output = (residual / RMS(residual)) * weight

By fusing these, we eliminate one redundant global-memory round-trip per
transformer layer (2x per layer in a standard Llama/Mistral block).

Mathematical formulation
------------------------
Given:
    - X_residual  : (B*T, D) residual stream tensor
    - X_hidden    : (B*T, D) output of the previous sub-layer (e.g. attention)
    - W           : (D,)     learnable RMSNorm weight
    - eps         : float    numerical stability constant

Compute:
    residual_out  = X_residual + X_hidden
    rms           = sqrt( (1/D) * sum(residual_out^2) + eps )
    normed_out    = (residual_out / rms) * W

Casting modes (following Liger-Kernel / HuggingFace conventions):
    - 'llama' : only the inverse RMS (rstd) is computed in fp32
    - 'gemma' : everything is cast to fp32 before computation
    - 'none'  : no casting, operate in the input dtype throughout

Reference implementations:
    - Liger-Kernel: linkedin/Liger-Kernel (rms_norm.py, FusedAddRMSNorm PR #812)
    - Unsloth:      unslothai/unsloth (rms_layernorm.py)

Author : ADLS Group (Software Stream)
Date   : March 2026
"""

import torch
import triton
import triton.language as tl
from enum import Enum


# ---------------------------------------------------------------------------
# Casting mode enum (mirrors Liger-Kernel conventions)
# ---------------------------------------------------------------------------
class CastingMode(Enum):
    NONE = 0
    LLAMA = 1
    GEMMA = 2


_STR_TO_CASTING_MODE = {
    "none": CastingMode.NONE,
    "llama": CastingMode.LLAMA,
    "gemma": CastingMode.GEMMA,
}


# ---------------------------------------------------------------------------
# Triton forward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_add_rmsnorm_fwd_kernel(
    # Pointers
    X_residual_ptr,   # (n_rows, n_cols) residual stream
    X_hidden_ptr,     # (n_rows, n_cols) sub-layer output
    Weight_ptr,       # (n_cols,)        RMSNorm weight
    Normed_out_ptr,   # (n_rows, n_cols) normalised output
    Residual_out_ptr, # (n_rows, n_cols) updated residual
    RSTD_ptr,         # (n_rows,)        cached 1/RMS per row (for backward)
    # Dimensions
    n_cols,
    # Hyperparameters
    eps,
    offset,           # weight offset (e.g. 1.0 for Gemma)
    # Compile-time constants
    CASTING_MODE: tl.constexpr,  # 0=none, 1=llama, 2=gemma
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program instance processes one row of the (B*T, D) tensor.
    We tile along the hidden dimension D with BLOCK_SIZE.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # ---- Pointers for this row ----
    residual_row_ptr = X_residual_ptr + row_idx * n_cols + col_offsets
    hidden_row_ptr = X_hidden_ptr + row_idx * n_cols + col_offsets

    # ---- Load inputs ----
    X_res = tl.load(residual_row_ptr, mask=mask, other=0.0)
    X_hid = tl.load(hidden_row_ptr, mask=mask, other=0.0)

    # ---- Fused residual addition ----
    residual = X_res + X_hid

    # ---- Compute RMS ----
    if CASTING_MODE == 2:  # gemma: cast everything to fp32
        residual_fp32 = residual.to(tl.float32)
        mean_sq = tl.sum(residual_fp32 * residual_fp32, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(mean_sq + eps)
        # Normalise in fp32
        W = tl.load(Weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        normed = residual_fp32 * rstd * (W + offset)
        # Cast back to original dtype
        normed = normed.to(residual.dtype)
    elif CASTING_MODE == 1:  # llama: only rstd in fp32
        residual_fp32 = residual.to(tl.float32)
        mean_sq = tl.sum(residual_fp32 * residual_fp32, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(mean_sq + eps)
        # Normalise: keep residual in original dtype, rstd is fp32
        W = tl.load(Weight_ptr + col_offsets, mask=mask, other=0.0)
        normed = residual * rstd.to(residual.dtype) * (W + offset)
    else:  # none: still accumulate reduction in fp32 for numerical stability
        residual_fp32 = residual.to(tl.float32)
        mean_sq = tl.sum(residual_fp32 * residual_fp32, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(mean_sq + eps)
        W = tl.load(Weight_ptr + col_offsets, mask=mask, other=0.0)
        normed = residual * rstd.to(residual.dtype) * (W + offset)

    # ---- Store outputs ----
    normed_out_row_ptr = Normed_out_ptr + row_idx * n_cols + col_offsets
    residual_out_row_ptr = Residual_out_ptr + row_idx * n_cols + col_offsets

    tl.store(normed_out_row_ptr, normed, mask=mask)
    tl.store(residual_out_row_ptr, residual, mask=mask)

    # Cache rstd for backward pass
    tl.store(RSTD_ptr + row_idx, rstd)


# ---------------------------------------------------------------------------
# Triton backward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_add_rmsnorm_bwd_kernel(
    # Pointers
    dNormed_ptr,       # (n_rows, n_cols) grad w.r.t. normed output
    dResidual_ptr,     # (n_rows, n_cols) grad w.r.t. residual output (downstream)
    Residual_ptr,      # (n_rows, n_cols) saved residual = X_residual + X_hidden
    Weight_ptr,        # (n_cols,)        RMSNorm weight
    RSTD_ptr,          # (n_rows,)        saved 1/RMS
    # Output gradient pointers
    dX_residual_ptr,   # (n_rows, n_cols) grad flowing back to residual input
    dX_hidden_ptr,     # (n_rows, n_cols) grad flowing back to hidden input
    dWeight_partial_ptr,  # (n_rows, n_cols) partial dW per row
    # Dimensions
    n_cols,
    # Hyperparameters
    offset,
    # Compile-time constants
    CASTING_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass for fused add + RMSNorm.

    RMSNorm backward:
        Let r = residual, rstd = 1/RMS(r), w = Weight + offset
        normed = r * rstd * w

        dL/dr = rstd * (dNormed * w) - (rstd^3 / n_cols) * sum(dNormed * w * r) * r

    Gradient through the addition (residual = X_res + X_hid):
        dL/dX_res = dL/dr + dL/d(residual_out)
        dL/dX_hid = dL/dr + dL/d(residual_out)
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # ---- Load saved values ----
    res_row_ptr = Residual_ptr + row_idx * n_cols + col_offsets
    R = tl.load(res_row_ptr, mask=mask, other=0.0)
    rstd = tl.load(RSTD_ptr + row_idx)
    W = tl.load(Weight_ptr + col_offsets, mask=mask, other=0.0)

    # ---- Load incoming gradients ----
    dNormed = tl.load(dNormed_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    dResidual_downstream = tl.load(
        dResidual_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0
    )

    # ---- RMSNorm backward ----
    w_eff = W + offset

    # All casting modes compute the backward in fp32 for numerical stability.
    # The forward kernel differentiates casting behaviour; the backward always
    # upcasts to fp32 because the reduction (dot product) and rstd^3 term are
    # highly sensitive to rounding in half-precision dtypes.
    R_fp32 = R.to(tl.float32)
    dNormed_fp32 = dNormed.to(tl.float32)
    w_eff_fp32 = w_eff.to(tl.float32)

    m = dNormed_fp32 * w_eff_fp32
    dot_mr = tl.sum(m * R_fp32, axis=0)
    dR = (rstd * m) - (rstd * rstd * rstd / n_cols) * dot_mr * R_fp32
    dR = dR.to(R.dtype)

    dW_partial = (dNormed_fp32 * R_fp32 * rstd).to(R.dtype)

    # ---- Combine: gradient through the addition ----
    total_grad = dR + dResidual_downstream

    # ---- Store gradients ----
    tl.store(dX_residual_ptr + row_idx * n_cols + col_offsets, total_grad, mask=mask)
    tl.store(dX_hidden_ptr + row_idx * n_cols + col_offsets, total_grad, mask=mask)
    tl.store(dWeight_partial_ptr + row_idx * n_cols + col_offsets, dW_partial, mask=mask)


# ---------------------------------------------------------------------------
# Autograd function wrapper
# ---------------------------------------------------------------------------
class FusedAddRMSNorm(torch.autograd.Function):
    """
    torch.autograd.Function wrapping the fused Triton kernels.

    Forward:
        normed_out, residual_out = FusedAddRMSNorm.apply(
            X_residual, X_hidden, weight, eps, offset, casting_mode
        )

    Backward:
        Computes gradients for X_residual, X_hidden, and weight.
    """

    @staticmethod
    def forward(ctx, X_residual, X_hidden, weight, eps=1e-6, offset=0.0, casting_mode="llama"):
        assert X_residual.shape == X_hidden.shape, (
            f"Shape mismatch: X_residual {X_residual.shape} vs X_hidden {X_hidden.shape}"
        )
        assert X_residual.shape[-1] == weight.shape[0], (
            f"Hidden dim mismatch: X {X_residual.shape[-1]} vs W {weight.shape[0]}"
        )
        assert X_residual.is_contiguous() and X_hidden.is_contiguous(), (
            "Input tensors must be contiguous"
        )

        casting_mode_enum = _STR_TO_CASTING_MODE.get(casting_mode, CastingMode.LLAMA)
        casting_mode_int = casting_mode_enum.value

        # Flatten to 2D
        orig_shape = X_residual.shape
        X_residual_2d = X_residual.view(-1, orig_shape[-1])
        X_hidden_2d = X_hidden.view(-1, orig_shape[-1])
        n_rows, n_cols = X_residual_2d.shape

        # Allocate outputs
        normed_out = torch.empty_like(X_residual_2d)
        residual_out = torch.empty_like(X_residual_2d)

        rstd_dtype = torch.float32 if casting_mode_int in (1, 2) else X_residual.dtype
        RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X_residual.device)

        # Block size: next power of 2 >= n_cols
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        if BLOCK_SIZE > 65536:
            raise ValueError(f"Hidden dim {n_cols} too large for single-row tiling (max 65536)")

        num_warps = min(max(BLOCK_SIZE // 256, 1), 16)

        # Launch forward kernel
        _fused_add_rmsnorm_fwd_kernel[(n_rows,)](
            X_residual_2d, X_hidden_2d, weight,
            normed_out, residual_out, RSTD,
            n_cols, eps, offset,
            CASTING_MODE=casting_mode_int,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(residual_out, weight, RSTD)
        ctx.n_cols = n_cols
        ctx.offset = offset
        ctx.casting_mode_int = casting_mode_int
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.orig_shape = orig_shape

        return normed_out.view(orig_shape), residual_out.view(orig_shape)

    @staticmethod
    def backward(ctx, dNormed, dResidual_out):
        residual_out, weight, RSTD = ctx.saved_tensors
        n_cols = ctx.n_cols
        offset = ctx.offset
        casting_mode_int = ctx.casting_mode_int
        BLOCK_SIZE = ctx.BLOCK_SIZE
        num_warps = ctx.num_warps
        orig_shape = ctx.orig_shape

        dNormed_2d = dNormed.contiguous().view(-1, n_cols)
        dResidual_2d = dResidual_out.contiguous().view(-1, n_cols)
        residual_2d = residual_out.view(-1, n_cols)
        n_rows = dNormed_2d.shape[0]

        dX_residual = torch.empty_like(residual_2d)
        dX_hidden = torch.empty_like(residual_2d)
        dWeight_partial = torch.empty_like(residual_2d)

        _fused_add_rmsnorm_bwd_kernel[(n_rows,)](
            dNormed_2d, dResidual_2d, residual_2d, weight, RSTD,
            dX_residual, dX_hidden, dWeight_partial,
            n_cols, offset,
            CASTING_MODE=casting_mode_int,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        # Reduce dWeight across rows (second-level reduction in PyTorch)
        dWeight = dWeight_partial.sum(dim=0)

        return (
            dX_residual.view(orig_shape),
            dX_hidden.view(orig_shape),
            dWeight,
            None, None, None,  # eps, offset, casting_mode
        )


# ---------------------------------------------------------------------------
# nn.Module wrapper (drop-in for MASE transform pass)
# ---------------------------------------------------------------------------
class FusedAddRMSNormModule(torch.nn.Module):
    """
    nn.Module wrapping the fused add + RMSNorm operation.

    Designed as a drop-in replacement that a MASE transform pass can swap in
    where it detects the pattern:
        residual = residual + hidden_states
        normed   = rmsnorm(residual, weight)

    Args:
        hidden_size (int):    dimension of the hidden states (D)
        eps (float):          epsilon for numerical stability
        offset (float):       weight offset, e.g. 1.0 for Gemma
        casting_mode (str):   'llama', 'gemma', or 'none'
    """

    def __init__(self, hidden_size, eps=1e-6, offset=0.0, casting_mode="llama"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.offset = offset
        self.casting_mode = casting_mode

    def forward(self, X_residual, X_hidden):
        return FusedAddRMSNorm.apply(
            X_residual, X_hidden, self.weight,
            self.eps, self.offset, self.casting_mode
        )

    def extra_repr(self):
        return (
            f"hidden_size={self.weight.shape[0]}, eps={self.eps}, "
            f"offset={self.offset}, casting_mode='{self.casting_mode}'"
        )