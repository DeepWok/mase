"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 02:17:08
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 02:17:08
"""

import contextlib
import logging
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import truncnorm
from torch import Tensor, nn
from torch.autograd import grad
from torch.nn.modules.utils import _pair
from torch.types import Device, _size

__all__ = [
    "toeplitz",
    "im2col_2d",
]


def shift(v: Tensor, f: float = 1) -> Tensor:
    return torch.cat((f * v[..., -1:], v[..., :-1]), dim=-1)


def Krylov(linear_map: Callable, v: Tensor, n: Optional[int] = None) -> Tensor:
    if n is None:
        n = v.size(-1)
    cols = [v]
    for _ in range(n - 1):
        v = linear_map(v)
        cols.append(v)
    return torch.stack(cols, dim=-2)


def circulant(eigens: Tensor) -> Tensor:
    circ = Krylov(shift, eigens).transpose(-1, -2)
    return circ


@lru_cache(maxsize=4)
def _get_toeplitz_indices(n: int, device: Device) -> Tensor:
    # cached toeplitz indices. avoid repeatedly generate the indices.
    indices = circulant(torch.arange(n, device=device))
    return indices


def toeplitz(col: Tensor) -> Tensor:
    """
    Efficient Toeplitz matrix generation from the first column. The column vector must in the last dimension. Batch generation is supported. Suitable for AutoGrad. Circulant matrix multiplication is ~4x faster than rfft-based implementation!\\
    @col {torch.Tensor} (Batched) column vectors.\\
    return out {torch.Tensor} (Batched) circulant matrices
    """
    n = col.size(-1)
    indices = _get_toeplitz_indices(n, device=col.device)
    return col[..., indices]


def im2col_2d(
    W: Optional[Tensor] = None,
    X: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    w_size: Optional[_size] = None,
) -> Tuple[Tensor, Tensor, int, int]:
    if W is not None:
        W_col = W.view(W.size(0), -1)
    else:
        W_col = None

    if X is not None:
        n_filters, d_filter, h_filter, w_filter = W.size() if W is not None else w_size
        n_x, d_x, h_x, w_x = X.size()

        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(
            X.view(1, -1, h_x, w_x),
            h_filter,
            dilation=1,
            padding=padding,
            stride=stride,
        ).view(n_x, -1, h_out * w_out)
        X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
    else:
        X_col, h_out, w_out = None, None, None

    return W_col, X_col, h_out, w_out


def complex_mult(X: Tensor, Y: Tensor) -> Tensor:
    """Complex-valued element-wise multiplication

    Args:
        X (Tensor): Real tensor with last dim of 2 or complex tensor
        Y (Tensor): Real tensor with last dim of 2 or complex tensor

    Returns:
        Tensor: tensor with the same type as input
    """
    if not torch.is_complex(X) and not torch.is_complex(Y):
        assert (
            X.shape[-1] == 2 and Y.shape[-1] == 2
        ), "Last dimension of real-valued tensor must be 2"
        if hasattr(torch, "view_as_complex"):
            return torch.view_as_real(
                torch.view_as_complex(X) * torch.view_as_complex(Y)
            )
        else:
            return torch.stack(
                (
                    X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
                    X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0],
                ),
                dim=-1,
            )
    else:
        return X.mul(Y)


def polar_to_complex(mag: Tensor, angle: Tensor) -> Tensor:
    # magnitude and angle to real and imag
    if angle is None:
        return real_to_complex(angle)
    if mag is None:
        if isinstance(angle, torch.Tensor):
            x = torch.stack([angle.cos(), angle.sin()], dim=-1)
        elif isinstance(angle, np.ndarray):
            x = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        else:
            raise NotImplementedError
    else:
        if isinstance(angle, torch.Tensor):
            x = torch.stack([mag * angle.cos(), mag * angle.sin()], dim=-1)
        elif isinstance(angle, np.ndarray):
            x = np.stack([mag * np.cos(angle), mag * np.sin(angle)], axis=-1)
        else:
            raise NotImplementedError
    return x


@lru_cache(maxsize=4)
def _polynomial_order_base(order: int, device: Device) -> Tensor:
    return torch.arange(order - 1, -1, -1, device=device)


def polynomial(x: Tensor | np.ndarray, coeff: Tensor | np.ndarray) -> Tensor:
    """calculate polynomial function of x given coefficient coeff

    Args:
        x (Tensor): input tensor
        coeff (Tensor): Tensor of shape [n], where n is the degree of polynomial. Orders: [n, n-1, ..., 2, 1, constant]

    Returns:
        Tensor: output tensor coeff[0]*x^n + coeff[1]*x^{n-1} + ... + coeff[n-1]*x + coeff[n]
    """
    # xs = [x]
    # for i in range(2, coeff.size(0)):
    #     xs.append(xs[-1]*x)
    # xs.reverse()
    # x = torch.stack(xs, dim=-1)

    # Deprecated implementation
    # x = torch.stack([x**i for i in range(coeff.size(0) - 1, 0, -1)], dim=-1)
    # out = (x * coeff[:-1]).sum(dim=-1) + coeff[-1].data.item()
    # return out

    ### x^n, x^{n-1}, ..., x^2, x, 1
    order = coeff.shape[0]  # n+1
    if isinstance(x, Tensor):
        ## torch from highest order to constant
        x = x[..., None].expand([-1] * x.dim() + [order])
        order_base = _polynomial_order_base(order, x.device)
        return x.pow(order_base).matmul(coeff)
    elif isinstance(x, np.ndarray):
        ## numpy polyval from constant to higher order
        return np.polynomial.polynomial.polyval(x, coeff[::-1])
    else:
        raise NotImplementedError
