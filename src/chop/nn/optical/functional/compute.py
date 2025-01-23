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

from .torch_train import set_torch_deterministic

__all__ = [
    "shift",
    "Krylov",
    "circulant",
    "toeplitz",
    "complex_circulant",
    "complex_mult",
    "expi",
    "complex_matvec_mult",
    "complex_matmul",
    "real_to_complex",
    "get_complex_magnitude",
    "get_complex_energy",
    "complex_to_polar",
    "polar_to_complex",
    "absclamp",
    "absclamp_",
    "im2col_2d",
    "check_identity_matrix",
    "check_unitary_matrix",
    "check_equal_tensor",
    "batch_diag",
    "batch_eye_cpu",
    "batch_eye",
    "merge_chunks",
    "partition_chunks",
    "clip_by_std",
    "percentile",
    "gen_boolean_mask_cpu",
    "gen_boolean_mask",
    "fftshift_cpu",
    "ifftshift_cpu",
    "gen_gaussian_noise",
    "gen_gaussian_filter2d_cpu",
    "gen_gaussian_filter2d",
    "add_gaussian_noise_cpu",
    "add_gaussian_noise",
    "add_gaussian_noise_",
    "circulant_multiply",
    "calc_diagonal_hessian",
    "calc_jacobian",
    "polynomial",
    "gaussian",
    "lowrank_decompose",
    "get_conv2d_flops",
    "interp1d",
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


def complex_circulant(eigens: Tensor) -> Tensor:
    circ = Krylov(shift, eigens).transpose(-1, -2)
    return circ


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


def complex_matvec_mult(W: Tensor, X: Tensor) -> Tensor:
    return torch.sum(complex_mult(W, X.unsqueeze(0).repeat(W.size(0), 1, 1)), dim=1)


def complex_matmul(X: Tensor, Y: Tensor) -> Tensor:
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, "Last dimension must be 2"
    if torch.__version__ >= "1.8" or (
        torch.__version__ >= "1.7" and X.shape[:-3] == Y.shape[:-3]
    ):
        return torch.view_as_real(
            torch.matmul(torch.view_as_complex(X), torch.view_as_complex(Y))
        )

    return torch.stack(
        [
            X[..., 0].matmul(Y[..., 0]) - X[..., 1].matmul(Y[..., 1]),
            X[..., 0].matmul(Y[..., 1]) + X[..., 1].matmul(Y[..., 0]),
        ],
        dim=-1,
    )


def expi(x: Tensor) -> Tensor:
    if torch.__version__ >= "1.8" or (
        torch.__version__ >= "1.7" and not x.requires_grad
    ):
        return torch.exp(1j * x)
    else:
        return x.cos().type(torch.cfloat) + 1j * x.sin().type(torch.cfloat)


def real_to_complex(x: Tensor) -> Tensor:
    if torch.__version__ < "1.7":
        return torch.stack((x, torch.zeros_like(x).to(x.device)), dim=-1)
    else:
        return torch.view_as_real(x.to(torch.complex64))


def get_complex_magnitude(x: Tensor) -> Tensor:
    assert x.size(-1) == 2, "[E] Input must be complex Tensor"
    return torch.sqrt(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1])


def complex_to_polar(x: Tensor) -> Tensor:
    # real and imag to magnitude and angle
    if isinstance(x, torch.Tensor):
        mag = x.norm(p=2, dim=-1)
        angle = torch.view_as_complex(x).angle()
        x = torch.stack([mag, angle], dim=-1)
    elif isinstance(x, np.ndarray):
        x = x.astype(np.complex64)
        mag = np.abs(x)
        angle = np.angle(x)
        x = np.stack([mag, angle], axis=-1)
    else:
        raise NotImplementedError
    return x


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


def get_complex_energy(x: Tensor) -> Tensor:
    assert x.size(-1) == 2, "[E] Input must be complex Tensor"
    return x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1]


def absclamp(
    x: Tensor, min: Optional[float] = None, max: Optional[float] = None
) -> Tensor:
    if isinstance(x, torch.Tensor):
        mag = x.norm(p=2, dim=-1).clamp(min=min, max=max)
        angle = torch.view_as_complex(x).angle()
        x = polar_to_complex(mag, angle)
    elif isinstance(x, np.ndarray):
        x = x.astype(np.complex64)
        mag = np.clip(np.abs(x), a_min=min, a_max=max)
        angle = np.angle(x)
        x = polar_to_complex(mag, angle)
    else:
        raise NotImplementedError
    return x


def absclamp_(
    x: Tensor, min: Optional[float] = None, max: Optional[float] = None
) -> Tensor:
    if isinstance(x, torch.Tensor):
        y = torch.view_as_complex(x)
        mag = y.abs().clamp(min=min, max=max)
        angle = y.angle()
        x.data.copy_(polar_to_complex(mag, angle))
    elif isinstance(x, np.ndarray):
        y = x.astype(np.complex64)
        mag = np.clip(np.abs(y), a_min=min, a_max=max)
        angle = np.angle(y)
        x[:] = polar_to_complex(mag, angle)
    else:
        raise NotImplementedError
    return x


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


def check_identity_matrix(W: Tensor) -> bool:
    if isinstance(W, np.ndarray):
        W_numpy = W.copy().astype(np.float64)
    elif isinstance(W, torch.Tensor):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    return (W_numpy.shape[0] == W_numpy.shape[1]) and np.allclose(
        W_numpy, np.eye(W_numpy.shape[0])
    )


def check_unitary_matrix(W: Tensor) -> bool:
    if isinstance(W, np.ndarray):
        W_numpy = W.copy().astype(np.float64)
    elif isinstance(W, torch.Tensor):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    M = np.dot(W_numpy, W_numpy.T)
    # print(M)
    return check_identity_matrix(M)


def check_equal_tensor(W1: Tensor, W2: Tensor) -> bool:
    if isinstance(W1, np.ndarray):
        W1_numpy = W1.copy().astype(np.float64)
    elif isinstance(W1, torch.Tensor):
        W1_numpy = W1.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    if isinstance(W2, np.ndarray):
        W2_numpy = W2.copy().astype(np.float64)
    elif isinstance(W2, torch.Tensor):
        W2_numpy = W2.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    return (W1_numpy.shape == W2_numpy.shape) and np.allclose(W1_numpy, W2_numpy)


def batch_diag(x: Tensor) -> Tensor:
    # x[..., N, N] -> [..., N]
    assert (
        len(x.shape) >= 2
    ), f"At least 2-D array/tensor is expected, but got shape {x.shape}"
    if isinstance(x, np.ndarray):
        size = list(x.shape)
        x = x.reshape(size[:-2] + [size[-2] * size[-1]])
        x = x[..., :: size[-1] + 1]
    elif isinstance(x, torch.Tensor):
        size = list(x.size())
        x = x.flatten(-2, -1)
        x = x[..., :: size[-1] + 1]
    else:
        raise NotImplementedError
    return x


def batch_eye_cpu(N: int, batch_shape: List[int], dtype: np.dtype) -> np.ndarray:
    x = np.zeros(list(batch_shape) + [N, N], dtype=dtype)
    x.reshape(-1, N * N)[..., :: N + 1] = 1
    return x


def batch_eye(
    N: int,
    batch_shape: List[int],
    dtype: torch.dtype,
    device: Device = torch.device("cuda"),
) -> torch.Tensor:
    x = torch.zeros(list(batch_shape) + [N, N], dtype=dtype, device=device)
    x.view(-1, N * N)[..., :: N + 1] = 1
    return x


def merge_chunks(x: Tensor, complex: bool = False) -> Tensor:
    """Merge a chunked/blocked tensors into a 2D matrix

    Args:
        x (Tensor): Tensor of shape [h1, w1, h2, w2, ...., hk, wk] if complex=False; [h1, w1, h2, w2, ...., hk, wk, 2] if complex=True
        complex (bool, optional): True if the tensor x has a last dimension with size 2 for real/imag representation. Defaults to False.

    Returns:
        Tensor: [h1*h2*...*hk, w1*w2*...*wk] or [h1*h2*...*hk, w1*w2*...*wk, 2]
    """
    if isinstance(x, torch.Tensor):
        permute = torch.permute
    elif isinstance(x, np.ndarray):
        permute = np.transpose
    else:
        raise NotImplementedError

    if not complex:
        dim = len(x.shape)
        x = permute(x, list(range(0, dim, 2)) + list(range(1, dim + 1, 2)))
        x = x.reshape(np.prod([x.shape[i] for i in range(dim // 2)]), -1)
    else:
        dim = len(x.shape) - 1
        x = permute(x, list(range(0, dim, 2)) + list(range(1, dim + 1, 2) + [dim]))
        x = x.reshape(np.prod([x.shape[i] for i in range(dim // 2)]), -1, 2)

    return x


def partition_chunks(
    x: Tensor, out_shape: int | Tuple[int, ...], complex: bool = False
) -> Tensor:
    """Partition a tensor into square chunks, similar to Rearrange in einops

    Args:
        x (Tensor): 2D tensor of shape [h1*h2*...*hk, w1*w2*...*wk] or 3D tensor of shape [h1*h2*...*hk, w1*w2*...*wk, 2] if complex=True
        out_shape (Tuple[int]): output blocked shape (h1, w1, h2, w2, ...); Do not include the last dimension even if complex=True
        complex (bool, optional): whether x is complex tensor. Defaults to False.

    Returns:
        [Tensor]: Tensor of shape [h1, w1, h2, w2, ...., hk, wk] or [h1, w1, h2, w2, ...., hk, wk, 2] if complex=True
    """
    if complex:
        assert len(x.shape) == 3
    x_shape = (np.prod(out_shape[::2]), np.prod(out_shape[1::2]))
    if isinstance(x, torch.Tensor):
        permute = torch.permute
        pad_fn = lambda x, padding: torch.nn.functional.pad(x[None, None], padding)[
            0, 0
        ]
        is_tensor = True
    elif isinstance(x, np.ndarray):
        permute = np.transpose
        pad_fn = np.pad
        is_tensor = False
    else:
        raise NotImplementedError

    if x_shape != x.shape[:2]:
        ## if x cannot be partitioned into out_shape, we need to pad it
        if is_tensor:
            ## torch from the last dim
            padding = (0, x_shape[1] - x.shape[1], 0, x_shape[0] - x.shape[0])
            if complex:
                padding = (0, 0) + padding
        else:
            ## np from the first dim
            padding = ((0, x_shape[0] - x.shape[0]), (0, x_shape[1] - x.shape[1]))
            if complex:
                padding = padding + (0, 0)

        x = pad_fn(x, padding)

    in_shape = list(out_shape[::2]) + list(out_shape[1::2])
    permute_shape = np.arange(len(out_shape)).reshape(2, -1).T.reshape(-1).tolist()
    if complex:
        in_shape.append(2)
        permute_shape.append(len(permute_shape))
    x = x.reshape(in_shape)  # [h1, h2, ..., hk, w1, w2, ..., wk]

    x = permute(x, permute_shape)  # [h1, w1, h2, w2, ...., hk, wk]

    return x


def clip_by_std(x: Tensor, n_std_neg: float = 3.0, n_std_pos: float = 3.0) -> Tensor:
    if isinstance(x, np.ndarray):
        std = np.std(x)
        mean = np.mean(x)
        out = np.clip(x, a_min=mean - n_std_neg * std, a_max=mean + n_std_pos * std)
    elif isinstance(x, torch.Tensor):
        std = x.data.std()
        mean = x.data.mean()
        out = x.clamp(min=mean - n_std_neg * std, max=mean + n_std_pos * std)
    else:
        raise NotImplementedError
    return out


def percentile(t: Tensor, q: float) -> Tensor:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    if isinstance(t, torch.Tensor):
        k = 1 + round(0.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
    elif isinstance(t, np.ndarray):
        result = np.percentile(t, q=q)
    else:
        raise NotImplementedError
    return result


def gen_boolean_mask_cpu(size: _size, true_prob: float) -> np.ndarray:
    assert 0 <= true_prob <= 1, "[E] Wrong probability for True"
    return np.random.choice(a=[False, True], size=size, p=[1 - true_prob, true_prob])


def gen_boolean_mask(
    size: _size,
    true_prob: float,
    random_state: Optional[int] = None,
    device: Device = torch.device("cuda"),
) -> Tensor:
    assert 0 <= true_prob <= 1, "[E] Wrong probability for True"
    if true_prob > 1 - 1e-9:
        return torch.ones(size, device=device, dtype=torch.bool)
    elif true_prob < 1e-9:
        return torch.zeros(size, device=device, dtype=torch.bool)
    if random_state is not None:
        with torch.random.fork_rng():
            torch.random.manual_seed(random_state)
            return torch.empty(size, dtype=torch.bool, device=device).bernoulli_(
                true_prob
            )
    else:
        return torch.empty(size, dtype=torch.bool, device=device).bernoulli_(true_prob)


def fftshift_cpu(
    x: Union[Tensor, np.ndarray], batched: bool = True, dim: Optional[Tuple[int]] = None
) -> Union[Tensor, np.ndarray]:
    if isinstance(x, np.ndarray):
        if dim is None:
            if batched:
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.fftshift(x, axes=dim)
    elif isinstance(x, torch.Tensor):
        device = x.device
        x = x.cpu().detach().numpy()
        if dim is None:
            if batched:
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.fftshift(x, axes=dim)
        out = torch.from_numpy(out).to(device)
    return out


def ifftshift_cpu(
    x: Union[Tensor, np.ndarray], batched: bool = True, dim: Optional[Tuple[int]] = None
) -> Union[Tensor, np.ndarray]:
    if isinstance(x, np.ndarray):
        if dim is None:
            if batched:
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.ifftshift(x, axes=dim)
    elif isinstance(x, torch.Tensor):
        device = x.device
        x = x.cpu().detach().numpy()
        if dim is None:
            if batched:
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.ifftshift(x, axes=dim)
        out = torch.from_numpy(out).to(device)
    return out


def gen_gaussian_noise(
    W: Union[Tensor, np.ndarray],
    noise_mean: float = 0.0,
    noise_std: float = 0.002,
    trunc_range: Tuple = (),
    random_state: Optional[int] = None,
) -> Union[Tensor, np.ndarray]:
    if random_state is not None:
        set_torch_deterministic(random_state)
    if isinstance(W, np.ndarray):
        if not trunc_range:
            noises = np.random.normal(noise_mean, noise_std, W.shape)
        else:
            a = (trunc_range[0] - noise_mean) / noise_std
            b = (trunc_range[1] - noise_mean) / noise_std
            noises = truncnorm.rvs(
                a, b, loc=noise_mean, scale=noise_std, size=W.shape, random_state=None
            )
    elif isinstance(W, torch.Tensor):
        if not trunc_range:
            noises = torch.zeros_like(W).normal_(mean=noise_mean, std=noise_std)
        else:
            size = W.shape
            tmp = W.new_empty(size + (4,)).normal_()
            a = (trunc_range[0] - noise_mean) / noise_std
            b = (trunc_range[1] - noise_mean) / noise_std
            valid = (tmp < b) & (tmp > a)
            ind = valid.max(-1, keepdim=True)[1]
            noises = tmp.gather(-1, ind).squeeze(-1).mul_(noise_std).add_(noise_mean)
            # noises = truncated_normal(W, mean=noise_mean, std=noise_std, a=trunc_range[0], b=trunc_range[1])
    else:
        assert 0, logging.error(
            f"Array type not supported, must be numpy.ndarray or torch.Tensor, but got {type(W)}"
        )
    return noises


def gen_gaussian_filter2d_cpu(size: int = 3, std: float = 0.286) -> np.ndarray:
    assert (
        size % 2 == 1
    ), f"Gaussian filter can only be odd size, but size={size} is given."
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 / np.square(std) * (np.square(xx) + np.square(yy)))
    kernel = kernel / np.sum(kernel)
    kernel[size // 2, size // 2] = 1
    return kernel


def gen_gaussian_filter2d(
    size: int = 3,
    std: float = 0.286,
    center_one: bool = True,
    device: Device = torch.device("cuda"),
) -> Tensor:
    assert (
        size % 2 == 1
    ), f"Gaussian filter can only be odd size, but size={size} is given."
    if std > 1e-8:
        ax = torch.linspace(
            -(size - 1) / 2.0,
            (size - 1) / 2.0,
            size,
            dtype=torch.float32,
            device=device,
        )
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-0.5 / (std**2) * (xx.square() + yy.square()))
        kernel = kernel.div_(kernel.sum())
        if center_one:
            kernel[size // 2, size // 2] = 1
    else:
        kernel = torch.zeros(size, size, dtype=torch.float32, device=device)
        kernel[size // 2, size // 2] = 1

    return kernel


def add_gaussian_noise(
    W: Union[Tensor, np.ndarray],
    noise_mean: float = 0,
    noise_std: float = 0.002,
    trunc_range: Tuple = (),
    random_state: Optional[int] = None,
) -> Union[Tensor, np.ndarray]:
    noises = gen_gaussian_noise(
        W,
        noise_mean=noise_mean,
        noise_std=noise_std,
        trunc_range=trunc_range,
        random_state=random_state,
    )
    output = W + noises
    return output


def add_gaussian_noise_(
    W: Union[Tensor, np.ndarray],
    noise_mean: float = 0,
    noise_std: float = 0.002,
    trunc_range: Tuple = (),
    random_state: Optional[int] = None,
) -> Union[Tensor, np.ndarray]:
    noises = gen_gaussian_noise(
        W,
        noise_mean=noise_mean,
        noise_std=noise_std,
        trunc_range=trunc_range,
        random_state=random_state,
    )
    if isinstance(W, np.ndarray):
        W += noises
    elif isinstance(W, torch.Tensor):
        W.data += noises
    else:
        assert 0, logging.error(
            f"Array type not supported, must be numpy.ndarray or torch.Tensor, but got {type(W)}"
        )
    return W


def add_gaussian_noise_cpu(
    W: Union[Tensor, np.ndarray],
    noise_mean: float = 0,
    noise_std: float = 0.002,
    trunc_range: Tuple = (),
) -> Union[Tensor, np.ndarray]:
    if isinstance(W, np.ndarray):
        W_numpy = W.copy().astype(np.float64)
    elif isinstance(W, torch.Tensor):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    if not trunc_range:
        noises = np.random.normal(noise_mean, noise_std, W_numpy.shape)
    else:
        a = (trunc_range[0] - noise_mean) / noise_std
        b = (trunc_range[1] - noise_mean) / noise_std
        noises = truncnorm.rvs(
            a, b, loc=noise_mean, scale=noise_std, size=W_numpy.shape, random_state=None
        )
    return W_numpy + noises


def circulant_multiply(c: Tensor, x: Tensor) -> Tensor:
    """Multiply circulant matrix with first column c by x
    Parameters:
        c: (n, )
        x: (batch_size, n) or (n, )
    Return:
        prod: (batch_size, n) or (n, )
    """
    return torch.irfft(
        complex_mult(torch.rfft(c, 1), torch.rfft(x, 1)), 1, signal_sizes=(c.shape[-1],)
    )


def calc_diagonal_hessian(weight_dict, loss, model):
    model.zero_grad()
    hessian_dict = {}
    for name, weight in weight_dict.items():
        first_gradient = grad(loss, weight, create_graph=True)[0]
        second_gradient = grad(first_gradient.sum(), weight, create_graph=True)[0]
        hessian_dict[name] = second_gradient.clone()
    model.zero_grad()
    return hessian_dict


def calc_jacobian(
    weight_dict: Dict[str, Tensor], loss: Callable, model: nn.Module
) -> Dict[str, Tensor]:
    model.zero_grad()
    jacobian_dict = {}
    for name, weight in weight_dict.items():
        first_gradient = grad(loss, weight, create_graph=True)[0]
        jacobian_dict[name] = first_gradient.clone()
    model.zero_grad()
    return jacobian_dict


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


def gaussian(x: Tensor, coeff: Tensor) -> Tensor:
    # coeff : [n, 3], includes a, b, c
    ## a * exp(-((x-b)/c)^2) + ...
    size = x.size()
    x = x.view(-1).unsqueeze(0)
    x = (
        (coeff[:, 0:1] * torch.exp(-((x - coeff[:, 1:2]) / coeff[:, 2:3]).square()))
        .sum(dim=0)
        .view(size)
    )
    return x


def lowrank_decompose(
    x: Tensor,
    r: int,
    u_ortho: bool = False,
    out_u: Optional[Tensor] = None,
    out_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """low rank decomposition on x. x ~ uv.

    Args:
        x (Tensor): tensor to decomplse
        r (int): rank
        u_ortho (bool, optional): whether u is orthogonal matrix. Defaults to False.
        out_u (Optional[Tensor], optional): output buffer for u. Defaults to None.
        out_v (Optional[Tensor], optional): output buffer for v. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: [description]
    """
    ### x [..., m, n]
    # r rank
    u, s, v = x.data.svd(some=True)
    v = v.transpose(-2, -1).contiguous()
    u = u[..., :, :r]
    s = s[..., :r]
    v = v[..., :r, :]
    if u_ortho == False:
        u.mul_(s.unsqueeze(-2))
    else:
        v.mul_(s.unsqueeze(-1))
    if out_u is not None:
        out_u.data.copy_(u)
    if out_v is not None:
        out_v.data.copy_(v)
    return u, v


def get_conv2d_flops(
    input_shape: _size,
    conv_filter: _size,
    stride: _pair = (1, 1),
    padding: _pair = (1, 1),
) -> float:
    # input_shape = (4, 3,300,300) # Format:(batch, channels, rows,cols)
    # conv_filter = (64,3,3,3)  # Format: (num_filters, channels, rows, cols)
    # stride = (1, 1) in (height, width)
    # padding = (1, 1) in (height, width)
    if type(stride) not in {list, tuple}:
        stride = [stride, stride]
    if type(padding) not in {list, tuple}:
        padding = [padding, padding]
    n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length
    # general defination for number of flops (n: multiplications and n-1: additions)
    flops_per_instance = n + 1

    num_instances_per_filter = (
        (input_shape[2] - conv_filter[2] + 2 * padding[0]) / stride[0]
    ) + 1  # for rows
    # multiplying with cols
    num_instances_per_filter *= (
        (input_shape[3] - conv_filter[3] + 2 * padding[1]) / stride[1]
    ) + 1

    flops_per_filter = num_instances_per_filter * flops_per_instance
    # multiply with number of filters adn batch
    total_flops_per_layer = flops_per_filter * conv_filter[0] * input_shape[0]
    return total_flops_per_layer


class Interp1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, xnew, out=None):
        """
        Batched Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`. Any point exceeds the border of [xmin, xmax]
        will be filled with 0 and no grad.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        https://github.com/aliutkus/torchinterp1d

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.

        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
            assert len(vec.shape) <= 2, "interp1d: all inputs must be " "at most 2-D."
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, "All parameters must be on the same device."
        device = device[0]

        # Checking for the dimensions
        assert v["x"].shape[1] == v["y"].shape[1] and (
            v["x"].shape[0] == v["y"].shape[0]
            or v["x"].shape[0] == 1
            or v["y"].shape[0] == 1
        ), (
            "x and y must have the same number of columns, and either "
            "the same number of row or one of them having only one "
            "row."
        )

        reshaped_xnew = False
        if (
            (v["x"].shape[0] == 1)
            and (v["y"].shape[0] == 1)
            and (v["xnew"].shape[0] > 1)
        ):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v["xnew"].shape
            v["xnew"] = v["xnew"].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v["x"].shape[0], v["xnew"].shape[0])
        shape_ynew = (D, v["xnew"].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0] * shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v["xnew"].shape[0] == 1:
            v["xnew"] = v["xnew"].expand(v["x"].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes
        torch.searchsorted(
            v["x"].contiguous().squeeze(), v["xnew"].contiguous(), out=ind
        )

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v["x"].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ["x", "y", "xnew"]:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [
                    None,
                ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat["slopes"] = is_flat["x"]
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v["slopes"] = (v["y"][:, 1:] - v["y"][:, :-1]) / (
                eps + (v["x"][:, 1:] - v["x"][:, :-1])
            )

            # now build the linear interpolation
            ynew = sel("y") + sel("slopes") * (v["xnew"] - sel("x"))

            mask = (v["xnew"] > v["x"][:, -1:]) | (
                v["xnew"] < v["x"][:, :1]
            )  # exceed left/right border
            ynew = ynew.masked_fill(mask, 0)

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
            ctx.saved_tensors[0],
            [i for i in inputs if i is not None],
            grad_out,
            retain_graph=True,
        )
        result = [
            None,
        ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


def interp1d(x: Tensor, y: Tensor, xnew: Tensor, out: Tensor | None = None) -> Tensor:
    """numpy.interp for pytorch. Only 1D

    Args:
        x (Tensor): input vector x coordinates
        y (Tensor): input vector y coordinates
        xnew (Tensor): new x coordinates to be interpolated
        out (Tensor, optional): output tensor. Defaults to None.

    Returns:
        Tensor: interpolated y coordinates
    """
    return Interp1d.apply(x, y, xnew, out)
