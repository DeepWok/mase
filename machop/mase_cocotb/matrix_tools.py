from copy import copy
import logging

import torch
from torch import Tensor

from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.utils import sign_extend_t, signed_to_unsigned, floor_rounding

from .utils import batched

logger = logging.getLogger("matrix_tools")
logger.setLevel(logging.DEBUG)


def split_matrix(x: Tensor, total_dim0, total_dim1, compute_dim0, compute_dim1):
    depth_dim0 = total_dim0 // compute_dim0
    depth_dim1 = total_dim1 // compute_dim1

    l = list()
    for i in range(depth_dim1):
        for j in range(depth_dim0):
            block = x[
                i * compute_dim1 : (i + 1) * compute_dim1,
                j * compute_dim0 : (j + 1) * compute_dim0,
            ]
            block_flatten = block.flatten().tolist()
            block_flatten.reverse()
            l.append(block_flatten)
    return l


def rebuild_matrix(
    x: list[list[int]], total_dim0, total_dim1, compute_dim0, compute_dim1
):
    depth_dim0 = total_dim0 // compute_dim0
    depth_dim1 = total_dim1 // compute_dim1
    assert len(x) == depth_dim0 * depth_dim1, "Not enough matrix blocks!"

    arr = torch.zeros(size=(total_dim1, total_dim0))
    block_ind = 0
    for i in range(depth_dim1):
        for j in range(depth_dim0):
            block = copy(x[block_ind])
            block.reverse()
            block = torch.tensor(block, dtype=torch.int32)
            block = block.reshape((compute_dim1, compute_dim0))
            arr[
                i * compute_dim1 : (i + 1) * compute_dim1,
                j * compute_dim0 : (j + 1) * compute_dim0,
            ] = block
            block_ind += 1
    return arr.int()


def gen_random_matrix_input(
    total_dim0, total_dim1, compute_dim0, compute_dim1, width, frac_width
):
    x = (torch.rand(size=(total_dim1, total_dim0)) - 0.5) * 2
    x *= 2 ** (width - frac_width - 1)
    x = quantize_to_int(x, width, frac_width)
    return split_matrix(x, total_dim0, total_dim1, compute_dim0, compute_dim1)


def matrix_mult_model(
    a_total_dim0,
    a_total_dim1,
    a_compute_dim0,
    a_compute_dim1,
    b_total_dim0,
    b_total_dim1,
    b_compute_dim0,
    b_compute_dim1,
    c_total_dim0,
    c_total_dim1,
    c_compute_dim0,
    c_compute_dim1,
    a_width,
    a_frac_width,
    b_width,
    b_frac_width,
    out_width,
    out_frac_width,
    out_symmetric,
    a_input,
    b_input,
    debug=False,
):
    A = rebuild_matrix(
        a_input, a_total_dim0, a_total_dim1, a_compute_dim0, a_compute_dim1
    )
    B = rebuild_matrix(
        b_input, b_total_dim0, b_total_dim1, b_compute_dim0, b_compute_dim1
    )
    A_signed = sign_extend_t(A, a_width)
    B_signed = sign_extend_t(B, b_width)
    C_signed = torch.matmul(A_signed, B_signed)

    if debug:
        logger.debug("Matrix A")
        logger.debug(A_signed)
        logger.debug("Matrix B")
        logger.debug(B_signed)
        logger.debug("Matrix C")
        logger.debug(C_signed)

    # Floor rounding
    acc_frac_width = a_frac_width + b_frac_width
    C_signed = floor_rounding(C_signed, acc_frac_width, out_frac_width)

    # Do clamp
    min_val = -(2 ** (out_width - 1)) + 1 if out_symmetric else -(2 ** (out_width - 1))
    max_val = (2 ** (out_width - 1)) - 1
    C_clamped = torch.clamp(C_signed, min_val, max_val)

    if debug:
        logger.debug("Matrix C (clamp)")
        logger.debug(C_clamped)

    # Changed to unsigned number
    C_unsigned_rep = signed_to_unsigned(C_clamped, out_width)

    if debug:
        logger.debug("Matrix C (clamp -> unsigned)")
        logger.debug(C_unsigned_rep)

    # Split into lists of ints
    return split_matrix(
        C_unsigned_rep, c_total_dim0, c_total_dim1, c_compute_dim0, c_compute_dim1
    )
