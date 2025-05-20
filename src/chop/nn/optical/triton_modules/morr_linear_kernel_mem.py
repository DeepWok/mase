import os
# os.environ["TRITON_INTERPRET"] = "1"

import torch
from torch import Tensor
import triton
import triton.language as tl
import pdb

from .dtype import TORCH_DTYPE_TO_TRITON
PACKAGE_NAME = "mase_triton"
from ..utils import (
    toeplitz,
    input_quantize_fn,
    weight_quantize_fn,
    mrr_roundtrip_phase_to_tr_func
)
from .quantize import _input_quantize_fn, _weight_quantize_fn


def _get_autotune_configs():
    configs = []
    for _M in [1, 2, 4, 8]:
        for _P in [1, 2, 4, 8]:
            for _Q in [1, 2, 4, 8]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_SIZE_M": _M,
                            "BLOCK_SIZE_P": _P,
                            "BLOCK_SIZE_Q": _Q,
                            # "BLOCK_SIZE_K1": 4,
                            "BLOCK_SIZE_K2": 1,
                        },
                        num_stages=3,
                        num_warps=8,
                    )
                )
    return configs

@triton.jit
def _mrr_roundtrip_phase_to_tr_func(
    x: tl.tensor,
    a: tl.constexpr = 0.8,
    r: tl.constexpr = 0.9,
    intensity: tl.constexpr = False,
):
    """
    Applies a round-trip phase correction to the input tensor.
    """
    c1 = -2.0 * a * r
    c2 = a * a + r * r
    c3 = 1.0 + r * r * a * a - a * a - r * r

    cos_x = tl.cos(x)
    numerator = cos_x * c1 + c2
    denominator = numerator + c3
    x = numerator / denominator
    if not intensity:
        x = tl.sqrt(x)
    return x

# @triton.autotune(
#     configs= [
#         triton.Config(
#         {
#             "BLOCK_SIZE_M": 1,
#             "BLOCK_SIZE_P": 1,
#             "BLOCK_SIZE_Q": 1,
#             # "BLOCK_SIZE_K1": 2,
#             "BLOCK_SIZE_K2": 1,
#         },
#         num_stages=3,
#         num_warps=8,
#     ),],
#     key=["M", "P", "Q", "K"],
# )
@triton.autotune(
    configs = _get_autotune_configs(),
    key=["M", "P", "Q", "K"],
)
@triton.jit
def morr_propagate_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    b_ptr,
    M,
    P,
    Q,
    K,
    grid_dim_q,
    grid_dim_p,
    miniblock,
    crosstalk_factor,
    phase_noise_std,
    mrr_a,
    mrr_r,
    in_bit,
    w_bit,
    seed,
    # stride
    stride_wm, stride_wp, stride_wq, stride_wk1, stride_wk2,
    stride_xm, stride_xp, stride_xq, stride_xk1, stride_xk2,
    stride_bm, stride_bp, stride_bq, stride_bk1,
    stride_om, stride_op, stride_oq, stride_ok1, stride_ok2,
    finegrain_drop_mask,
    ENABLE_PHASE_NOISE: tl.constexpr,
    ENABLE_THERMAL_CROSSTALK: tl.constexpr,
    TRAINABLE_MORR_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_K2: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):

    # Program ID for block-based processing
    # each program is assigned GROUP_SIZE_MPQ * [1, 1, miniblock, 1] block
    pid = tl.program_id(axis=0)
    # number of blocks (each program needs to handle) along M, P, Q dimension
    pnum_m = grid_dim_p * grid_dim_q
    pnum_p = grid_dim_p // BLOCK_SIZE_P
    pnum_q = grid_dim_q // BLOCK_SIZE_Q
    # block dimension of current program
    pid_m = pid // (pnum_q * pnum_p)
    pid_p = (pid // pnum_q) % pnum_p
    pid_q = pid % pnum_q

    # starting element's m, p, q coordinates in the global tensor
    start_m = pid_m * BLOCK_SIZE_M
    start_p = pid_p * BLOCK_SIZE_P
    start_q = pid_q * BLOCK_SIZE_Q
    
    # w [1, p, q, k, 1] -> toeplitz [1, p, q, k, k]
    offs_wm = tl.arange(0, 1)
    offs_wp = pid_p * BLOCK_SIZE_P + tl.arange(0, 1)
    offs_wq = pid_q * BLOCK_SIZE_Q + tl.arange(0, 1)
    offs_wk1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_wk2 = tl.arange(0, BLOCK_SIZE_K1)

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, 1)
    offs_xp = tl.arange(0, 1)
    offs_xq = pid_q * BLOCK_SIZE_Q + tl.arange(0, 1)
    offs_xk1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_xk2 = tl.arange(0, BLOCK_SIZE_K2)
    # morr_bias: [1, p, q, 1]
    offs_bm = tl.arange(0, 1)
    offs_bp = pid_p * BLOCK_SIZE_P + tl.arange(0, 1)
    offs_bq = pid_q * BLOCK_SIZE_Q + tl.arange(0, 1)
    offs_bk1 = tl.arange(0, 1)

    w_ptrs = w_ptr + (
        offs_wm[:, None, None, None, None] * stride_wm
        + offs_wp[None, :, None, None, None] * stride_wp
        + offs_wq[None, None, :, None, None] * stride_wq
        + offs_wk1[None, None, None, :, None] * stride_wk1
        + offs_wk2[None, None, None, None, :] * stride_wk2
    )
    x_ptrs = x_ptr + (
        offs_xm[:, None, None, None, None] * stride_xm
        + offs_xp[None, :, None, None, None] * stride_xp
        + offs_xq[None, None, :, None, None] * stride_xq
        + offs_xk1[None, None, None, :, None] * stride_xk1
        + offs_xk2[None, None, None, None, :] * stride_xk2
    )
    b_ptrs = b_ptr + (
        offs_bm[:, None, None, None, None] * stride_bm
        + offs_bp[None, :, None, None, None] * stride_bp
        + offs_bq[None, None, :, None, None] * stride_bq
        + offs_bk1[None, None, None, :, None] * stride_bk1
    )


    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P, BLOCK_SIZE_Q, BLOCK_SIZE_K1, BLOCK_SIZE_K2), dtype=tl.float32)
    m_indices = tl.arange(0, BLOCK_SIZE_M)[:, None, None, None, None]
    p_indices = tl.arange(0, BLOCK_SIZE_P)[None, :, None, None, None]
    q_indices = tl.arange(0, BLOCK_SIZE_Q)[None, None, :, None, None]

    for m_local in range(BLOCK_SIZE_M):
        m = start_m + m_local
        for p_local in range(BLOCK_SIZE_P):
            p = start_p + p_local
            for q_local in range(BLOCK_SIZE_Q):
                q = start_q + q_local

                w_mask = (p < P) & (q < Q)      
                x_mask = (m < M) & (q < Q)
                b_mask = (p < P) & (q < Q)

                w = tl.load(w_ptrs, mask=w_mask, other=0.0)
                x = tl.load(x_ptrs, mask=x_mask, other=0.0)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
   

                w = w.reshape(BLOCK_SIZE_K1, BLOCK_SIZE_K1) # [1, 1, 1, k, k] -> [k, k]
                x = x.reshape(BLOCK_SIZE_K1, BLOCK_SIZE_K2) # [1, 1, 1, k, 1] -> [k, 1]
                

                x = x * x  # input_modulator()
                # ----- propagate_morr() -----

                # apply thermal crosstalk noise
                if ENABLE_THERMAL_CROSSTALK:
                    w = w * crosstalk_factor
                
                # MatMals
                # TODO: tl.dot requires 16*16 matrix at least, this is a workaround
                x = tl.trans(x)
                x = tl.broadcast_to(x, (BLOCK_SIZE_K1, BLOCK_SIZE_K1))
                x = tl.sum(w * x, axis=1)
                x = tl.reshape(x, (BLOCK_SIZE_K1, BLOCK_SIZE_K2))

                # apply phase noise
                if ENABLE_PHASE_NOISE:
                    block_start = pid * BLOCK_SIZE_K1 * BLOCK_SIZE_K2
                    offs = tl.reshape(block_start + tl.arange(0, BLOCK_SIZE_K1 * BLOCK_SIZE_K2) , (BLOCK_SIZE_K1, BLOCK_SIZE_K2))
                    noise = tl.randn(seed, offs) * phase_noise_std
                    x = x + noise

                # add trainable bias
                b = b.reshape(1, 1)
                # pdb.set_trace()
                if TRAINABLE_MORR_BIAS:
                    x = x - b
                
                # mrr_roundtrip_phase_to_tr
                x = _mrr_roundtrip_phase_to_tr_func(x, mrr_a, mrr_r, intensity=True)

                # store the value in acc using mask
                res = x
                condition_mask = (m_indices == m_local) & (p_indices == p_local) & (q_indices == q_local)
                res = res[None, None, None, :, :]
                acc = tl.where(condition_mask, res, acc)    

                # propagate pointer along Q dimension
                w_ptrs += stride_wq
                x_ptrs += stride_xq
                b_ptrs += stride_bq
            
            # Q loop end
            # reset pointer along Q dimension
            w_ptrs -= stride_wq * (BLOCK_SIZE_Q)
            x_ptrs -= stride_xq * (BLOCK_SIZE_Q)
            b_ptrs -= stride_bq * (BLOCK_SIZE_Q)
            # propagate pointer along P dimension
            w_ptrs += stride_wp
            b_ptrs += stride_bp
            # x_ptrs += stride_xp # x has P dimension = 1
        
        # P loop end
        # reset pointer along P dimension
        w_ptrs -= stride_wp * (BLOCK_SIZE_P)
        b_ptrs -= stride_bp * (BLOCK_SIZE_P)
        # x_ptrs -= stride_xp * (BLOCK_SIZE_P + 1) # x has P  dimension = 1、

        # propagate pointer along M dimension
        # w_ptrs += stride_wp # weight has M dimension = 1
        x_ptrs += stride_xm


    out = acc.to(INPUT_DTYPE)
    out = out.reshape(BLOCK_SIZE_M, BLOCK_SIZE_P, BLOCK_SIZE_Q, BLOCK_SIZE_K1) # [1, 1, q, k, 1] -> [1, 1, q, k]

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_op = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    offs_oq = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_ok1 = tl.arange(0, BLOCK_SIZE_K1)
    # offs_ok2 = tl.arange(0, BLOCK_SIZE_K2)
    o_ptrs = o_ptr + (
        stride_om * offs_om[:, None, None, None]
        + stride_op * offs_op[None, :, None, None]
        + stride_oq * offs_oq[None, None, :, None]
        + stride_ok1 * offs_ok1[None, None, None, :]
    )

    m_valid = offs_om[:, None, None, None] < M
    p_valid = offs_op[None, :, None, None] < P
    q_valid = offs_oq[None, None, :, None] < Q
    k_valid = offs_ok1[None, None, None, :] < K   # K == BLOCK_SIZE_K1
    o_mask = m_valid & p_valid & q_valid & k_valid
    tl.store(o_ptrs, out, mask=o_mask)

@torch.library.custom_op(
    f"{PACKAGE_NAME}::optical_morr_linear_linear_fn", mutates_args={},
)
def morr_linear_fn_mem(
    x: Tensor,
    weight: Tensor,
    morr_input_bias: Tensor,
    morr_output_scale: Tensor,
    bias: Tensor | None,
    morr_input_scale: Tensor,
    morr_bias: Tensor | None,
    grid_dim_x: int,
    grid_dim_y: int,
    miniblock: int,
    enable_thermal_crosstalk: bool,
    crosstalk_factor: float | None,
    enable_phase_noise: bool,
    phase_noise_std: float | None,
    trainable_morr_bias: bool,
    mrr_a: float,
    mrr_r: float,
    finegrain_drop_mask: Tensor | None,
    in_features: int,
    in_features_pad: int,
    out_features: int,
    out_features_pad: int,
    in_bit: int,
    w_bit: int,
    morr_fwhm: float,
    sigma_weight: float,
    trainable_morr_scale: bool,
    morr_scale: Tensor,
    weight_quant_gain: float | None = None,
    seed: int=42,
) -> tuple[Tensor, int, Tensor, Tensor, Tensor, Tensor, Tensor, float]:
    Device = x.device
    assert x.dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ), f"Unsupported dtype {x.dtype}"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert weight.dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ), f"Unsupported dtype {weight.dtype}"

    # Handle transformer vs non-transformer inputs
    ori_x_shape = x.shape
    is_transformer = len(ori_x_shape) == 3

    if is_transformer:
        in_B, in_N, in_D = x.shape
        M = in_B * in_N
        x = x.reshape(M, in_D)
    else:
        M = x.shape[0]

    # Get dimensions
    M, D = x.shape
    P, Q, K = weight.shape

    if in_features_pad > D:
        x_pad = torch.zeros(M, in_features_pad - D, device=Device, dtype=x.dtype)
        x = torch.cat([x, x_pad], dim=1)
    
    assert Q * K == in_features_pad, "input and weight dimension mismatch"
    assert P * K == out_features_pad, "weight and output dimension mismatch"

    # Quantize input
    if in_bit < 16:
        input_quantizer = input_quantize_fn(in_bit, device=Device)
        input_quantizer.set_bitwidth(in_bit)
        x = input_quantizer(x)
    
    # Build weight
    if w_bit < 16:
        weight_quantizer = weight_quantize_fn(w_bit, alg="dorefa_pos")
        weight_quantizer.set_bitwidth(w_bit)
        weight = weight_quantizer(weight)

        ## rescale weights after quantization can maintain the initialization distribution
        if weight_quant_gain is None:
            weight_quant_gain = sigma_weight / weight.data.std()
        if trainable_morr_scale:
            morr_scale = morr_scale * weight_quant_gain
        else:
            morr_scale = weight_quant_gain
        weight = weight.mul(
            morr_scale
        )  ### gain factor from Tanh used in quantization
        ### quantize learnable balancing factor
        morr_output_scale_quantizer = weight_quantize_fn(w_bit, alg="dorefa_sym")
        morr_output_scale = morr_output_scale_quantizer(morr_output_scale)
    else:
        weight = weight.abs()  # positive only
        morr_output_scale = (morr_output_scale - morr_output_scale.data.mean())
    
    if finegrain_drop_mask is not None:
        weight = weight.mul(finegrain_drop_mask.float())
    
    # differential balancing factor concatenation
    scale = morr_output_scale[..., :-1, :]
    scale_pad = morr_output_scale[..., -1:, :]
    if grid_dim_x % 2 == 0:
        # even blocks
        scale = torch.cat([scale, -scale], dim=2)  # [1, 1, q, 1]
    else:
        # odd blocks
        if grid_dim_x > 1:
            scale = torch.cat([morr_output_scale, -scale], dim=2)  # [1, 1, q, 1]
        else:
            scale = scale_pad  # [1, 1, q, 1]
    morr_output_scale = scale.squeeze(-1).unsqueeze(0)  # [1 ,1, 1, q]
    ctx_morr_output_scale = morr_output_scale.clone()

    # Reshape x and weight
    x = x.view(-1, grid_dim_x, miniblock)  # [M, q, k]
    x = x.unsqueeze(1).unsqueeze(-1) # [M, 1, q, k, 1]
    weight = toeplitz(weight).unsqueeze(0) # [p, q, k] -> [1, p, q, k, k]

    x_ctx = x.squeeze(-1).squeeze(1).clone() # [M, q, k]
    w_ctx = weight.clone()
    
    # Allocate output
    output = torch.empty((M, P, Q, K, 1), device=Device, dtype=x.dtype)
    # Launch the Triton kernel
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(P, meta["BLOCK_SIZE_P"]) * triton.cdiv(Q, meta["BLOCK_SIZE_Q"]),
    )
    morr_propagate_kernel[grid](
        x_ptr = x,
        w_ptr = weight,
        o_ptr = output,
        b_ptr = morr_bias,
        M=M,
        P=P,
        Q=Q,
        K=K,
        grid_dim_q=grid_dim_x,
        grid_dim_p=grid_dim_y,
        miniblock=miniblock,
        crosstalk_factor=crosstalk_factor,
        phase_noise_std=phase_noise_std,
        mrr_a=mrr_a,
        mrr_r=mrr_r,
        in_bit=in_bit,
        w_bit=w_bit,
        seed=seed,
        finegrain_drop_mask=finegrain_drop_mask,
        stride_wm=weight.stride(0),
        stride_wp=weight.stride(1),
        stride_wq=weight.stride(2),
        stride_wk1=weight.stride(3),
        stride_wk2=weight.stride(4),
        stride_xm=x.stride(0),
        stride_xp=x.stride(1),
        stride_xq=x.stride(2),
        stride_xk1=x.stride(3),
        stride_xk2=x.stride(4),
        stride_bm=morr_bias.stride(0) if morr_bias is not None else 0,
        stride_bp=morr_bias.stride(1) if morr_bias is not None else 0,
        stride_bq=morr_bias.stride(2) if morr_bias is not None else 0,
        stride_bk1=morr_bias.stride(3) if morr_bias is not None else 0,
        stride_om=output.stride(0),
        stride_op=output.stride(1),
        stride_oq=output.stride(2),
        stride_ok1=output.stride(3),
        stride_ok2=output.stride(4),
        ENABLE_THERMAL_CROSSTALK=enable_thermal_crosstalk,
        ENABLE_PHASE_NOISE=enable_phase_noise and phase_noise_std > 1e-4,
        TRAINABLE_MORR_BIAS = trainable_morr_bias,
        INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
        BLOCK_SIZE_K1=K,
    )

    # Apply output scale
    output = output.squeeze(-1)  # [m, p, q, k, 1] -> [m, p, q, k]
    ctx_x_scalematmul = output.clone() # record x input for matmul
    output = morr_output_scale.matmul(output)  # [1, 1, 1, q] x [bs, p, q, k] = [bs, p, 1, k]
    output = output.flatten(1) # [bs, p*k]

    # Trim output if needed
    if out_features < out_features_pad:
        output = output[:, :out_features]
    if bias is not None:
        output = output + bias.unsqueeze(0)
    # Reshape back for transformer
    if is_transformer:
        output = output.view(in_B, in_N, out_features)

    return output, seed, torch.abs(w_ctx), x_ctx, ctx_morr_output_scale, ctx_x_scalematmul, morr_scale.clone(), weight_quant_gain if weight_quant_gain is not None else 0.0



def _morr_linear_setup_context(ctx, inputs, output):
    """
    Save for backward only what the backward routine really needs.
    """
    (
        x,                       # 0  Tensor – input
        weight,                  # 1  Tensor – learnable weight
        morr_input_bias,         # 23 Tensor
        _,                       # 3 morr_output_scale (original)
        bias,                    # 4  Tensor | None – bias
        morr_input_scale,
        morr_bias,               # 2 Tensor | None
        grid_dim_x,              # 5  int
        grid_dim_y,              # 6  int
        miniblock,               # 7  int (== K)
        enable_thermal_crosstalk,# 8  bool
        crosstalk_factor,        # 9  float
        enable_phase_noise,      # 10  bool
        phase_noise_std,         # 11  float
        trainable_morr_bias,     # 12 bool
        mrr_a,                   # 13 float
        mrr_r,                   # 14 float
        finegrain_drop_mask,     # 15 Tensor | None
        in_features,             # 16 int
        in_features_pad,         # 17 int
        out_features,            # 18 int
        out_features_pad,        # 19 int
        in_bit,                  # 20 int
        w_bit,                   # 21 int
        morr_fwhm,               # 22 float
        sigma_weight,
        trainable_morr_scale, # bool
        _morr_scale,
        weight_quant_gain,
        seed,                    # 23 int
    ) = inputs

    output, seed, w_morr, x_modulator, morr_output_scale, x_scalematmul, morr_scale, _weight_quant_gain = output

    device, dtype = x.device, x.dtype

    # ----- Tensor meta-data that backward needs -----
    # Shapes
    M = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]
    P, Q, K = weight.shape
    tensor_shape = (M, P, Q, K)

    # mrr_para: para for mrr_roundtrip_phase_to_tr()
    # c1 = -2.0 * mrr_a * mrr_r
    # c2 = mrr_a * mrr_a + mrr_r * mrr_r
    # c3 = 1.0 + (mrr_r * mrr_r) * (mrr_a * mrr_a) - mrr_a * mrr_a - mrr_r * mrr_r
    # c4 = (mrr_a**2.0 - 1.0) * (mrr_r**2.0 - 1.0) * 2.0 * mrr_a * mrr_r                                       
    # intensity = True
    # mrr_para = (c1, c2, c3, c4, intensity)
    
    # # x_morr: x input of matmal in propagate_morr()
    # x_morr = x_modulator ** 2 # [m, q, k]
    # x_morr = x_morr.unsqueeze(1).unsqueeze(-1) # [m, 1, q, k, 1]

    # # x_mrr: x input of mrr_roundtrip_phase_to_tr()
    # x_mrr = w_morr.matmul(x_morr).squeeze(-1)
    # if enable_phase_noise and phase_noise_std > 1e-5:
    #     x_mrr = x_mrr + torch.zeros_like(x_mrr).normal_(0, phase_noise_std)
    # if trainable_morr_bias:
    #     x_mrr = x_mrr - morr_bias # morr_bias here is the detached one from forward

    # tanh_input_bias = torch.tanh(morr_input_bias.unsqueeze(0).unsqueeze(-1)) # Added from linear.py

    # 3. stash tensors 
    ctx.save_for_backward(
        x,                        # original input (stashing x for mem version, might need re-evaluation for pure mem-saving)
        weight,                   # original weight (stashing weight for mem version)
        bias if bias is not None else torch.tensor([], device=device, dtype=dtype),
        morr_output_scale,        # original morr_output_scale
        # x_mrr,                    # x input for mrr_roundtrip_phase_to_tr()
        # x_morr,
        # w_morr,                   # w input for propagate_morr() matmul
        # x_modulator,              # x input for input_modulator()
        morr_input_bias, 
        # x_scalematmul,
        # x_scalematmul,   # x input for morr_output_scale.matmul
        morr_input_scale, # morr input scale at input
        # morr_scale, # morr_scale after modification in build_weight()
        finegrain_drop_mask,
    )
    ctx.tensor_shape = tensor_shape
    # ctx.mrr_para = mrr_para
    ctx.in_features = in_features 
    ctx.in_features_pad = in_features_pad                     
    ctx.out_features = out_features     
    ctx.out_features_pad = out_features_pad
    ctx.morr_fwhm = morr_fwhm
    ctx.grid_dim_x = grid_dim_x
    ctx.grid_dim_y = grid_dim_y
    ctx.in_bit = in_bit
    ctx.w_bit = w_bit
    ctx.x_input_shape = x.shape
    ctx.device = x.device
    ctx.w_input_shape = weight.shape
    # ctx.morr_fwhm = morr_fwhm # Already exists
    ctx.enable_phase_noise = enable_phase_noise
    ctx.phase_noise_std = phase_noise_std
    ctx.trainable_morr_bias = trainable_morr_bias
    ctx.trainable_morr_scale = trainable_morr_scale
    ctx.weight_quant_gain = weight_quant_gain
    ctx.miniblock = miniblock
    ctx.crosstalk_factor = crosstalk_factor
    ctx.sigma_weight = sigma_weight
    ctx.enable_thermal_crosstalk = enable_thermal_crosstalk
    ctx.mrr_a = mrr_a
    ctx.mrr_r = mrr_r

def recompute_activations(
    ctx,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    morr_output_scale: Tensor,
    finegrain_drop_mask,
    morr_input_bias: Tensor,
    morr_input_scale: Tensor,
):
    """
    Recompute activations for morr_linear_fn.
    """
    Device = x.device
    Dtype = x.dtype

    ctx_morr_scale = None
    ctx_tanh_input_bias = None

    # Handle transformer vs non-transformer inputs
    ori_x_shape = x.shape
    is_transformer = len(ori_x_shape) == 3

    if is_transformer:
        in_B, in_N, in_D = x.shape
        M = in_B * in_N
        x = x.reshape(M, in_D)
    else:
        M = x.shape[0]

    # Get dimensions
    M, D = x.shape
    P, Q, K = weight.shape

    if ctx.in_features_pad > D:
        x_pad = torch.zeros(M, ctx.in_features_pad - D, device=Device, dtype=x.dtype)
        x = torch.cat([x, x_pad], dim=1)

    # Quantize input
    if ctx.in_bit < 16:
        input_quantizer = input_quantize_fn(ctx.in_bit, device=Device)
        input_quantizer.set_bitwidth(ctx.in_bit)
        x = input_quantizer(x)
    
    ################# Build weight #################
    if ctx.w_bit < 16:
        weight_quantizer = weight_quantize_fn(ctx.w_bit, alg="dorefa_pos")
        weight_quantizer.set_bitwidth(ctx.w_bit)
        weight = weight_quantizer(weight)

        # Calculate morr_scale
        if morr_input_scale is None:
            return None
        morr_scale = torch.sigmoid(morr_input_scale.unsqueeze(-1)) + 0.2  # [p, q, 1]

        ## rescale weights after quantization can maintain the initialization distribution
        weight_quant_gain = ctx.weight_quant_gain
        if weight_quant_gain is None:
            weight_quant_gain = ctx.sigma_weight / weight.data.std()
        if ctx.trainable_morr_scale:
            morr_scale = morr_scale * weight_quant_gain
        else:
            morr_scale = weight_quant_gain
        
        ctx_morr_scale = morr_scale.clone()
        weight = weight.mul(
            morr_scale
        )  ### gain factor from Tanh used in quantization
        ### quantize learnable balancing factor
        morr_output_scale_quantizer = weight_quantize_fn(ctx.w_bit, alg="dorefa_sym")
        morr_output_scale = morr_output_scale_quantizer(morr_output_scale)
    else:
        weight = weight.abs()  # positive only
        morr_output_scale = (morr_output_scale - morr_output_scale.data.mean())
    
    if finegrain_drop_mask is not None:
        weight = weight.mul(finegrain_drop_mask.float())
    
    # differential balancing factor concatenation
    scale = morr_output_scale[..., :-1, :]
    scale_pad = morr_output_scale[..., -1:, :]
    if ctx.grid_dim_x % 2 == 0:
        # even blocks
        scale = torch.cat([scale, -scale], dim=2)  # [1, 1, q, 1]
    else:
        # odd blocks
        if ctx.grid_dim_x > 1:
            scale = torch.cat([morr_output_scale, -scale], dim=2)  # [1, 1, q, 1]
        else:
            scale = scale_pad  # [1, 1, q, 1]
    morr_output_scale = scale.squeeze(-1).unsqueeze(0)  # [1 ,1, 1, q]
    ctx_morr_output_scale = morr_output_scale.clone()

    # Reshape x and weight
    x = x.view(-1, ctx.grid_dim_x, ctx.miniblock)  # [M, q, k]

    # input_modulator()
    ctx_x_modulator = x.clone()
    x = x ** 2
    

    ################# propagate_morr() #################
    if ctx.enable_thermal_crosstalk and ctx.crosstalk_factor > 1:
            weight = weight * ctx.crosstalk_factor
    weight = toeplitz(weight).unsqueeze(0)  # [1, p, q, k, k]
    x = x.unsqueeze(1).unsqueeze(-1)  # [bs, 1, q, k, 1]

    ctx_x_morr = x.clone()
    ctx_w_morr = weight.clone()
    x = weight.matmul(x).squeeze(-1)  # [bs, p, q, k]

    if ctx.enable_phase_noise and ctx.phase_noise_std > 1e-5:
        x = x + torch.zeros_like(x).normal_(0, ctx.phase_noise_std)

    if ctx.trainable_morr_bias:
        ctx_tanh_input_bias = torch.tanh(morr_input_bias.unsqueeze(0).unsqueeze(-1))
        morr_bias = ctx.morr_fwhm * ctx_tanh_input_bias
        x = x - morr_bias
    
    ctx_x_mrr = x.clone()
    
    mrr_roundtrip_phase_to_tr = mrr_roundtrip_phase_to_tr_func(a=ctx.mrr_a, r=ctx.mrr_r, intensity=True)
    x = mrr_roundtrip_phase_to_tr(x)

    ctx_x_scalematmul = x.clone()
    x = morr_output_scale.matmul(x) # [1, 1, 1, q] x [bs, p, q, k] = [bs, p, 1, k]
    x = x.flatten(1) # [bs, p*k]

    # ------------------------------------------------------

    # # Trim output if needed
    # if ctx.out_features < ctx.out_features_pad:
    #     output = output[:, :ctx.out_features]
    # if bias is not None:
    #     output = output + bias.unsqueeze(0)
    # # Reshape back for transformer
    # if is_transformer:
    #     output = output.view(in_B, in_N, ctx.out_features)

    return (
        # x, weight, bias, morr_output_scale,
        # output, 
        ctx_x_modulator, # x input for input_modulator()
        ctx_x_morr, # x input for propagate_morr() matmul
        ctx_w_morr, # w input for propagate_morr() matmul
        ctx_x_mrr, # x input for mrr_roundtrip_phase_to_tr()
        ctx_x_scalematmul, # x input for morr_output_scale.matmul
        ctx_tanh_input_bias, # input_bias after tanh()
        ctx_morr_scale, # morr_scale after modification in build_weight()
    )

def _morr_linear_backward(ctx, grad_output, *ignored):
    """
    Backward pass for morr_linear_fn.
    """
    (
        x, 
        weight, 
        bias,
        morr_output_scale,
        # x_mrr,
        # x_morr,
        # w_morr,
        # x_modulator,
        morr_input_bias,
        # x_scalematmul,
        morr_input_scale,
        # morr_scale,
        finegrain_drop_mask
    ) = ctx.saved_tensors

    M, P, Q, K  = ctx.tensor_shape
    # c1, c2, c3, c4, intensity = ctx.mrr_para
    in_features = ctx.in_features
    in_features_pad = ctx.in_features_pad
    out_features = ctx.out_features
    out_features_pad = ctx.out_features_pad
    x_input_shape = ctx.x_input_shape
    w_input_shape = ctx.w_input_shape
    DEVICE = ctx.device

    # --- calculate intermediate activation on the fly ---
    (
        x_modulator, # x input for input_modulator()
        x_morr, # x input for propagate_morr() matmul
        w_morr, # w input for propagate_morr() matmul
        x_mrr, # x input for mrr_roundtrip_phase_to_tr()
        x_scalematmul, # x input for morr_output_scale.matmul
        tanh_input_bias, # input_bias after tanh()
        morr_scale, # morr_scale after modificaiton in build_weight()
    ) = recompute_activations(
        ctx, 
        x, 
        weight, 
        bias, 
        morr_output_scale, 
        finegrain_drop_mask, 
        morr_input_bias, 
        morr_input_scale
    )

    # x_morr = (x_modulator ** 2).unsqueeze(1).unsqueeze(-1)  # [m, q, k] -> # [m, 1, q, k, 1]

    # tanh_input_bias = torch.tanh(morr_input_bias.unsqueeze(0).unsqueeze(-1))
    # morr_bias = ctx.morr_fwhm * tanh_input_bias

    # # x_mrr: x input of mrr_roundtrip_phase_to_tr()
    # x_mrr = w_morr.matmul(x_morr).squeeze(-1)
    # if ctx.enable_phase_noise and ctx.phase_noise_std > 1e-5:
    #     x_mrr = x_mrr + torch.zeros_like(x_mrr).normal_(0, ctx.phase_noise_std)
    # if ctx.trainable_morr_bias:
    #     x_mrr = x_mrr - morr_bias
    
    
    
    # ----- backward prop -----
    # Reshape
    grad_out = grad_output.view(
        x_input_shape[0], 
        w_input_shape[1], 
        w_input_shape[2], 
        -1
    )  # [M, P, Q, K]
    
    # ----- Gradient w.r.t input x -----
    if ctx.needs_input_grad[0]:
        # 1. reshape
        grad_out = grad_out.view(M, -1) # [m, out_features]

        if ctx.needs_input_grad[4] and bias:
            grad_bias = grad_out.sum(dim=0) # [out_features]
        else:
            grad_bias = None

        out_pad = torch.zeros(grad_out.shape[0], out_features_pad-out_features, device = DEVICE) # [m, out_features_pad - out_features]
        grad_out = torch.cat([grad_out, out_pad], dim=1) # [m * out_features_pad] = [m, p*k]

        # 2. x=x.flatten(1)
        # input: [m, p**k]
        grad_out = grad_out.view(M, P, 1, K) # [m, p, 1, k]

        # 3. x = morr_output_scale.matmul(x)  # [1, 1, 1, q] x [bs, p, q, k] = [bs, p, 1, k]
        # dL/d(morr_output_scale)
        if ctx.needs_input_grad[3]:
            grad_s = grad_out.matmul(x_scalematmul.transpose(-2, -1)) # [bs, p, 1, q]
            grad_s = grad_s.sum(dim=(0, 1)).unsqueeze(0).unsqueeze(1) # [1, 1, 1, q] 
            grad_s = grad_s.squeeze(0).unsqueeze(-1) # [1, 1, q, 1] gradient of scale

            t  = ctx.grid_dim_x // 2
            grad_scale = grad_s.new_zeros((1, 1, t+1, 1))

            if ctx.grid_dim_x % 2 == 0:
                grad_scale[..., :t, :] = grad_s[..., :t, :] - grad_s[..., t:, :]
            elif ctx.grid_dim_x == 1:
                grad_scale = grad_s
            else:
                grad_scale[..., :t, :] = grad_s[..., :t, :] - grad_s[..., t+1:, :]
                grad_scale[..., t:t+1, :] = grad_s[..., t:t+1, :]
              
        else:
            grad_scale = None
        
        # dL/dx
        grad_x = morr_output_scale.transpose(-2, -1).matmul(grad_out) # [bs, p, q, k]

        # 4. x = mrr_roundtrip_phase_to_tr(x)
        mrr_a, mrr_r = ctx.mrr_a, ctx.mrr_r
        c1 = -2.0 * mrr_a * mrr_r
        c2 = mrr_a * mrr_a + mrr_r * mrr_r
        c3 = 1.0 + (mrr_r * mrr_r) * (mrr_a * mrr_a) - mrr_a * mrr_a - mrr_r * mrr_r
        c4 = (mrr_a**2.0 - 1.0) * (mrr_r**2.0 - 1.0) * 2.0 * mrr_a * mrr_r                                       
        intensity = True
        denominator = x_mrr.cos().mul_(c1).add_(c2 + c3)
        if intensity:
            denominator.square_()
            numerator = x_mrr.sin().mul_(c4)
        else:
            numerator = x_mrr.sin().mul_(c4 / 2)
            denominator = (
                denominator.sub(1).pow_(1.5).mul_(denominator.sub(c3).sqrt_())
            )
        grad_x = numerator.div_(denominator).mul_(grad_x) # [bs, p, q, k]
        
        # 5. x += phase_noise and morr_bias
        if ctx.trainable_morr_bias and ctx.needs_input_grad[2]:
            grad_inputbias = - grad_x # [bs, p, q, k]
            grad_inputbias = grad_inputbias * ctx.morr_fwhm # [bs, p, q, k]
            grad_inputbias = grad_inputbias - tanh_input_bias * tanh_input_bias # [bs, p, q, k]
            grad_inputbias = grad_inputbias.sum(dim=(0, -1))
        else:
            grad_inputbias = None

        # 6. x = weight.matmul(x) [1, p, q, k, k] * [bs, 1, q, k, 1] = [bs, p, q, k, 1]
        grad_x = grad_x.unsqueeze(-1) # [bs, p, q, k, 1]
        grad_morr_matmul = grad_x     # stash for weight gradient
        
        # dL/dx
        grad_x = torch.matmul(w_morr.transpose(-1, -2), grad_x) # [1, p, q, k, k] x [bs, p, q, k, 1] = [bs, p, q, k, 1]
        grad_x = grad_x.sum(dim=1, keepdim=True) # [bs, p, q, k, 1] -> [bs, 1, q, k, 1]
        grad_x = grad_x.squeeze(-1).squeeze(1) # [bs, 1, q, k, 1] -> [bs, q, k]

        # 7. input modulator
        grad_x = grad_x * 2 * x_modulator # [bs, q, k]

        # 8. input reshape
        grad_x = grad_x.view(x_input_shape)
        grad_x = grad_x[:, :in_features]



    # ----- Gradient w.r.t weight -----
    if ctx.needs_input_grad[1]:
        
        # 0. gradient after x = weight.matmul(x)
        # grad_morr_matmul # [bs, p, q, k, 1]

        # 1. x = weight.matmul(x)
        grad_w = torch.matmul(grad_morr_matmul, x_morr.transpose(-1,-2)) # [bs,p,q,k,k]
        grad_w = grad_w.sum(dim=0, keepdim=True) # [1,p,q,k,k]

        # 2. weight = toeplitz(weight)
        k = grad_w.size(-1)
        row = torch.arange(k)[:, None]        # (k,1)
        col = torch.arange(k)[None, :]        # (1,k)
        idx = (row - col) & (k - 1) if (k & (k-1)) == 0 else (row - col + k) % k

        idx = idx.expand(grad_w.shape).to(DEVICE)
        buffer = torch.zeros_like(grad_w, device=DEVICE)
        buffer.scatter_add_(-2, idx, grad_w) # [1, p, q, k, k]
        grad_w = buffer.sum(dim=-1, keepdim=True).squeeze(0).squeeze(-1)

        # 3. build_weight()
        if finegrain_drop_mask is not None:
            grad_w = grad_w * finegrain_drop_mask.float()
        # morr_scale: [p, q, 1]
        grad_morr_input_scale = None
        if ctx.w_bit < 16:
            # grad w.r.t morr_scale 
            if ctx.needs_input_grad[5] & ctx.trainable_morr_scale:
                grad_morr_scale = (grad_w * weight).sum(dim=2, keepdim=True) # [p, q, 1]
                grad_morr_scale = grad_morr_scale * ctx.weight_quant_gain # [p, q, 1]
                # ∂L/∂self.morr_input_scale
                sigmoid_scale = torch.sigmoid(morr_input_scale)
                grad_morr_input_scale = (grad_morr_scale * sigmoid_scale * (1-sigmoid_scale)).squeeze(-1) # [p, q]

            # grad w.r.t weight
            grad_w = grad_w * morr_scale
        else:
            grad_w = grad_w * weight.sign()
    
    return (
        grad_x,               # ∂L/∂x
        grad_w,          # ∂L/∂w
        grad_inputbias, # ∂L/∂morr_input_bias
        grad_scale,  # ∂L/∂morr_output_scale
        grad_bias,        # ∂L/∂bias
        grad_morr_input_scale,
        None, None, None, None, None, None, None, None, None,
        None, None, None,
        None, None, None, None, None, None, None,
        None, None, None, None
    )


morr_linear_fn_mem.register_autograd(
    _morr_linear_backward, setup_context=_morr_linear_setup_context,
)