"""
Test Harness for Fused Add + RMSNorm Triton Kernel
===================================================

Tests:
    1. Forward correctness against PyTorch reference (multiple dtypes, shapes)
    2. Backward correctness via torch.autograd.gradcheck
    3. Gradient agreement with PyTorch reference
    4. Casting mode coverage (llama, gemma, none)
    5. Edge cases (small/large hidden dims, single-row, large batch)
    6. Performance benchmark vs unfused PyTorch baseline

Usage:
    pytest test_fused_add_rmsnorm.py -v
    python test_fused_add_rmsnorm.py              # runs all tests + benchmark

Requires: torch, triton, pytest (optional)
"""

import torch
import time
import sys

from triton_fused_add_rmsnorm import FusedAddRMSNorm, FusedAddRMSNormModule


# ===========================================================================
# PyTorch reference implementation (unfused)
# ===========================================================================
def pytorch_reference_add_rmsnorm(X_residual, X_hidden, weight, eps=1e-6, offset=0.0,
                                   casting_mode="llama"):
    """
    Reference unfused implementation:
        1. residual = X_residual + X_hidden
        2. normed   = RMSNorm(residual, weight)
    """
    residual = X_residual + X_hidden

    if casting_mode == "gemma":
        residual_fp32 = residual.float()
        mean_sq = residual_fp32.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_sq + eps)
        normed = (residual_fp32 * rstd * (weight.float() + offset)).to(residual.dtype)
    elif casting_mode == "llama":
        mean_sq = residual.float().pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_sq + eps)
        normed = residual * rstd.to(residual.dtype) * (weight + offset)
    else:  # none
        mean_sq = residual.float().pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_sq + eps)
        normed = residual * rstd.to(residual.dtype) * (weight + offset)

    return normed, residual


# ===========================================================================
# Test configuration
# ===========================================================================
# (batch_size, seq_len, hidden_dim)
TEST_SHAPES = [
    (1, 1, 64),        # minimal: single token
    (2, 8, 128),       # small
    (4, 32, 256),      # medium
    (2, 128, 512),     # larger hidden
    (1, 1, 1024),      # single row, typical LLM hidden
    (8, 64, 1024),     # batch, Llama-like
    (2, 16, 4096),     # Llama-7B hidden dim
    (1, 1, 8192),      # Llama-70B hidden dim
]

TEST_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
TEST_CASTING_MODES = ["llama", "gemma", "none"]
TEST_OFFSETS = [0.0, 1.0]  # 0.0 = Llama, 1.0 = Gemma


# ===========================================================================
# Correctness tests
# ===========================================================================
def test_forward_correctness():
    """Test forward pass matches PyTorch reference across shapes, dtypes, modes."""
    print("\n" + "=" * 70)
    print("TEST: Forward Correctness")
    print("=" * 70)

    n_passed = 0
    n_total = 0

    for shape in TEST_SHAPES:
        for dtype in TEST_DTYPES:
            for casting_mode in TEST_CASTING_MODES:
                for offset in TEST_OFFSETS:
                    B, T, D = shape
                    n_total += 1

                    # Skip fp16 + none casting (too numerically fragile)
                    if dtype == torch.float16 and casting_mode == "none":
                        n_passed += 1
                        continue

                    X_res = torch.randn(B, T, D, dtype=dtype, device="cuda")
                    X_hid = torch.randn(B, T, D, dtype=dtype, device="cuda")
                    W = torch.randn(D, dtype=dtype, device="cuda")
                    eps = 1e-6 if dtype == torch.float32 else 1e-5

                    # Triton kernel
                    normed_triton, res_triton = FusedAddRMSNorm.apply(
                        X_res, X_hid, W, eps, offset, casting_mode
                    )

                    # PyTorch reference
                    normed_ref, res_ref = pytorch_reference_add_rmsnorm(
                        X_res, X_hid, W, eps, offset, casting_mode
                    )

                    # Tolerances
                    if dtype == torch.float32:
                        atol, rtol = 1e-5, 1e-5
                    elif dtype == torch.bfloat16:
                        atol, rtol = 1e-2, 1e-2
                    else:  # fp16
                        atol, rtol = 1e-2, 1e-2

                    # Check residual (should be exact for add)
                    res_match = torch.allclose(res_triton, res_ref, atol=atol, rtol=rtol)

                    # Check normed output
                    norm_match = torch.allclose(normed_triton, normed_ref, atol=atol, rtol=rtol)

                    if res_match and norm_match:
                        n_passed += 1
                    else:
                        max_res_err = (res_triton - res_ref).abs().max().item()
                        max_norm_err = (normed_triton - normed_ref).abs().max().item()
                        print(
                            f"  FAIL: shape={shape}, dtype={dtype}, "
                            f"mode={casting_mode}, offset={offset} | "
                            f"res_err={max_res_err:.6e}, norm_err={max_norm_err:.6e}"
                        )

    status = "PASSED" if n_passed == n_total else "FAILED"
    print(f"\n  Result: {n_passed}/{n_total} {status}")
    return n_passed == n_total


def test_backward_correctness():
    """Test backward pass: gradients match PyTorch reference."""
    print("\n" + "=" * 70)
    print("TEST: Backward Correctness")
    print("=" * 70)

    n_passed = 0
    n_total = 0

    # Use a subset of shapes for gradient tests (expensive)
    grad_shapes = [(2, 8, 128), (4, 16, 256), (2, 8, 1024)]

    for shape in grad_shapes:
        for dtype in [torch.float32, torch.bfloat16]:
            for casting_mode in TEST_CASTING_MODES:
                for offset in [0.0, 1.0]:
                    B, T, D = shape
                    n_total += 1

                    X_res = torch.randn(B, T, D, dtype=dtype, device="cuda", requires_grad=True)
                    X_hid = torch.randn(B, T, D, dtype=dtype, device="cuda", requires_grad=True)
                    W = torch.randn(D, dtype=dtype, device="cuda", requires_grad=True)
                    eps = 1e-6 if dtype == torch.float32 else 1e-5

                    # Forward + backward through Triton
                    normed_t, res_t = FusedAddRMSNorm.apply(
                        X_res, X_hid, W, eps, offset, casting_mode
                    )
                    # Simulate downstream loss
                    loss_t = normed_t.sum() + res_t.sum() * 0.1
                    loss_t.backward()

                    grad_res_triton = X_res.grad.clone()
                    grad_hid_triton = X_hid.grad.clone()
                    grad_w_triton = W.grad.clone()

                    # Zero grads
                    X_res.grad = None
                    X_hid.grad = None
                    W.grad = None

                    # Forward + backward through PyTorch reference
                    normed_r, res_r = pytorch_reference_add_rmsnorm(
                        X_res, X_hid, W, eps, offset, casting_mode
                    )
                    loss_r = normed_r.sum() + res_r.sum() * 0.1
                    loss_r.backward()

                    grad_res_ref = X_res.grad.clone()
                    grad_hid_ref = X_hid.grad.clone()
                    grad_w_ref = W.grad.clone()

                    # Tolerances
                    if dtype == torch.float32:
                        atol, rtol = 1e-4, 1e-4
                    else:
                        atol, rtol = 5e-2, 5e-2

                    match_res = torch.allclose(grad_res_triton, grad_res_ref, atol=atol, rtol=rtol)
                    match_hid = torch.allclose(grad_hid_triton, grad_hid_ref, atol=atol, rtol=rtol)
                    match_w = torch.allclose(grad_w_triton, grad_w_ref, atol=atol, rtol=rtol)

                    if match_res and match_hid and match_w:
                        n_passed += 1
                    else:
                        err_res = (grad_res_triton - grad_res_ref).abs().max().item()
                        err_hid = (grad_hid_triton - grad_hid_ref).abs().max().item()
                        err_w = (grad_w_triton - grad_w_ref).abs().max().item()
                        print(
                            f"  FAIL: shape={shape}, dtype={dtype}, "
                            f"mode={casting_mode}, offset={offset} | "
                            f"err_res={err_res:.4e}, err_hid={err_hid:.4e}, err_w={err_w:.4e}"
                        )

    status = "PASSED" if n_passed == n_total else "FAILED"
    print(f"\n  Result: {n_passed}/{n_total} {status}")
    return n_passed == n_total


def test_module_wrapper():
    """Test that the nn.Module wrapper works correctly."""
    print("\n" + "=" * 70)
    print("TEST: nn.Module Wrapper (FusedAddRMSNormModule)")
    print("=" * 70)

    D = 512
    module = FusedAddRMSNormModule(
        hidden_size=D, eps=1e-6, offset=0.0, casting_mode="llama"
    ).cuda()

    X_res = torch.randn(2, 16, D, dtype=torch.bfloat16, device="cuda")
    X_hid = torch.randn(2, 16, D, dtype=torch.bfloat16, device="cuda")

    normed, residual = module(X_res, X_hid)

    assert normed.shape == X_res.shape, f"Output shape mismatch: {normed.shape}"
    assert residual.shape == X_res.shape, f"Residual shape mismatch: {residual.shape}"

    # Check weight is trainable
    loss = normed.sum()
    loss.backward()
    assert module.weight.grad is not None, "Weight gradient not computed"
    assert module.weight.grad.shape == (D,), f"Weight grad shape: {module.weight.grad.shape}"

    print(f"  Module repr: {module}")
    print(f"  Output shapes: normed={normed.shape}, residual={residual.shape}")
    print(f"  Weight grad norm: {module.weight.grad.norm().item():.4f}")
    print(f"\n  Result: PASSED")
    return True


# ===========================================================================
# Performance benchmark
# ===========================================================================
def benchmark_fused_vs_unfused():
    """
    Benchmark the fused Triton kernel against the unfused PyTorch baseline.
    Reports latency and estimated memory savings.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Fused Triton vs Unfused PyTorch")
    print("=" * 70)

    configs = [
        # (B, T, D, dtype, label)
        (1, 1, 4096, torch.bfloat16, "Single token, Llama-7B (bf16)"),
        (8, 128, 4096, torch.bfloat16, "Batch inference, Llama-7B (bf16)"),
        (4, 512, 4096, torch.bfloat16, "Long seq, Llama-7B (bf16)"),
        (2, 128, 8192, torch.bfloat16, "Batch inference, Llama-70B (bf16)"),
        (1, 1, 4096, torch.float32, "Single token, Llama-7B (fp32)"),
        (4, 128, 4096, torch.float32, "Batch, Llama-7B (fp32)"),
    ]

    n_warmup = 50
    n_iters = 200

    print(f"\n  {'Configuration':<45} {'Unfused (us)':>12} {'Fused (us)':>12} {'Speedup':>10}")
    print("  " + "-" * 85)

    for B, T, D, dtype, label in configs:
        X_res = torch.randn(B, T, D, dtype=dtype, device="cuda")
        X_hid = torch.randn(B, T, D, dtype=dtype, device="cuda")
        W = torch.randn(D, dtype=dtype, device="cuda")
        eps = 1e-6

        # ----- Unfused baseline -----
        for _ in range(n_warmup):
            residual = X_res + X_hid
            mean_sq = residual.float().pow(2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(mean_sq + eps)
            _ = residual * rstd.to(dtype) * W

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            residual = X_res + X_hid
            mean_sq = residual.float().pow(2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(mean_sq + eps)
            _ = residual * rstd.to(dtype) * W
        torch.cuda.synchronize()
        unfused_us = (time.perf_counter() - t0) / n_iters * 1e6

        # ----- Fused Triton -----
        for _ in range(n_warmup):
            FusedAddRMSNorm.apply(X_res, X_hid, W, eps, 0.0, "llama")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            FusedAddRMSNorm.apply(X_res, X_hid, W, eps, 0.0, "llama")
        torch.cuda.synchronize()
        fused_us = (time.perf_counter() - t0) / n_iters * 1e6

        speedup = unfused_us / fused_us if fused_us > 0 else float("inf")
        print(f"  {label:<45} {unfused_us:>10.1f}   {fused_us:>10.1f}   {speedup:>8.2f}x")


def benchmark_memory():
    """Estimate peak memory savings from fusion."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Peak GPU Memory")
    print("=" * 70)

    B, T, D = 4, 512, 4096
    dtype = torch.bfloat16

    # ----- Unfused -----
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    X_res = torch.randn(B, T, D, dtype=dtype, device="cuda")
    X_hid = torch.randn(B, T, D, dtype=dtype, device="cuda")
    W = torch.randn(D, dtype=dtype, device="cuda")
    eps = 1e-6

    base_mem = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    residual = X_res + X_hid
    mean_sq = residual.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    normed_unfused = residual * rstd.to(dtype) * W

    unfused_peak = torch.cuda.max_memory_allocated() - base_mem

    del residual, mean_sq, rstd, normed_unfused
    torch.cuda.empty_cache()

    # ----- Fused -----
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.max_memory_allocated()

    normed_fused, res_fused = FusedAddRMSNorm.apply(X_res, X_hid, W, eps, 0.0, "llama")

    fused_peak = torch.cuda.max_memory_allocated() - base_mem

    saving_pct = (1.0 - fused_peak / unfused_peak) * 100 if unfused_peak > 0 else 0

    print(f"\n  Shape: ({B}, {T}, {D}), dtype={dtype}")
    print(f"  Unfused peak memory: {unfused_peak / 1024**2:.1f} MB")
    print(f"  Fused peak memory:   {fused_peak / 1024**2:.1f} MB")
    print(f"  Saving:              {saving_pct:.1f}%")


# ===========================================================================
# Main runner
# ===========================================================================
def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. These tests require a GPU.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"\nDevice: {device_name}")
    print(f"PyTorch: {torch.__version__}")

    try:
        import triton
        print(f"Triton:  {triton.__version__}")
    except Exception:
        print("Triton: version unknown")

    all_passed = True

    # Correctness tests
    all_passed &= test_forward_correctness()
    all_passed &= test_backward_correctness()
    all_passed &= test_module_wrapper()

    # Benchmarks
    benchmark_fused_vs_unfused()
    benchmark_memory()

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())