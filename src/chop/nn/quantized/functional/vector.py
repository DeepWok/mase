"""Vector-datapath rounding for decode precision experiments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from chop.nn.quantizers._minifloat_mx import (
    MinifloatMeta,
    minifloat_quantizer_sim,
)


@dataclass(frozen=True)
class VectorFormat:
    """Plain floating-point format used by the vector datapath."""

    token: str
    exponent_bits: int
    fraction_bits: int
    is_bfloat16: bool = False


_VECTOR_FORMATS = {
    "FP_E3M2": VectorFormat("FP_E3M2", 3, 2),
    "FP_E2M3": VectorFormat("FP_E2M3", 2, 3),
    "FP_E6M5": VectorFormat("FP_E6M5", 6, 5),
    "FP_E5M6": VectorFormat("FP_E5M6", 5, 6),
    "FP_E4M7": VectorFormat("FP_E4M7", 4, 7),
    "FP_E8M5": VectorFormat("FP_E8M5", 8, 5),
    "BF16": VectorFormat("BF16", 8, 7, is_bfloat16=True),
}
VECTOR_FP_FORMATS: Mapping[str, VectorFormat] = MappingProxyType(_VECTOR_FORMATS)


def parse_vector_format(token: str) -> VectorFormat:
    """Resolve a canonical vector format token."""

    try:
        return VECTOR_FP_FORMATS[token.upper()]
    except KeyError as exc:
        valid = ", ".join(VECTOR_FP_FORMATS)
        raise ValueError(
            f"Unsupported vector format {token!r}; expected one of {valid}"
        ) from exc


@dataclass(frozen=True)
class VectorRoundingPolicy:
    """Round every stored vector-operation result to one plain FP format."""

    format: VectorFormat | None
    is_finite: bool = False
    round_mode: str = "rn"

    @property
    def enabled(self) -> bool:
        return self.format is not None

    @classmethod
    def disabled(cls) -> "VectorRoundingPolicy":
        return cls(format=None)

    @classmethod
    def from_token(
        cls,
        token: str,
        *,
        is_finite: bool = False,
        round_mode: str = "rn",
    ) -> "VectorRoundingPolicy":
        return cls(
            format=parse_vector_format(token),
            is_finite=is_finite,
            round_mode=round_mode,
        )

    @classmethod
    def from_config(cls, config: dict | None) -> "VectorRoundingPolicy":
        """Build a policy from a stage config or a canonical token."""

        cfg = config or {}
        if cfg.get("bypass", False):
            return cls.disabled()

        token = next(
            (
                cfg[key]
                for key in ("vector_format", "fp_format", "format")
                if isinstance(cfg.get(key), str)
            ),
            None,
        )
        if token is not None:
            return cls.from_token(
                token,
                is_finite=cfg.get("data_in_is_finite", False),
                round_mode=cfg.get("data_in_round_mode", "rn"),
            )

        exponent_bits = cfg.get("data_in_exponent_width")
        fraction_bits = cfg.get("data_in_frac_width")
        if exponent_bits is None or fraction_bits is None:
            raise ValueError(
                "Vector precision requires a canonical format token or both "
                "data_in_exponent_width and data_in_frac_width"
            )

        for vector_format in VECTOR_FP_FORMATS.values():
            if (
                not vector_format.is_bfloat16
                and vector_format.exponent_bits == exponent_bits
                and vector_format.fraction_bits == fraction_bits
            ):
                return cls(
                    vector_format,
                    is_finite=cfg.get("data_in_is_finite", False),
                    round_mode=cfg.get("data_in_round_mode", "rn"),
                )
        if (
            exponent_bits <= 0
            or fraction_bits <= 0
            or exponent_bits + fraction_bits >= 16
        ):
            raise ValueError(
                f"Invalid vector FP fields E{exponent_bits}M{fraction_bits}"
            )
        return cls(
            VectorFormat(
                f"FP_E{exponent_bits}M{fraction_bits}",
                exponent_bits,
                fraction_bits,
            ),
            is_finite=cfg.get("data_in_is_finite", False),
            round_mode=cfg.get("data_in_round_mode", "rn"),
        )

    def round(self, tensor: Tensor) -> Tensor:
        """Round one vector register or SRAM boundary."""

        if not self.enabled:
            return tensor
        if not tensor.is_floating_point():
            raise TypeError("Vector rounding requires a floating-point tensor")
        if self.format.is_bfloat16:
            return tensor.to(torch.bfloat16).to(tensor.dtype)
        return minifloat_quantizer_sim(
            tensor,
            minifloat_meta=MinifloatMeta(
                exp_bits=self.format.exponent_bits,
                frac_bits=self.format.fraction_bits,
                is_finite=self.is_finite,
                round_mode=self.round_mode,
            ),
            output_dtype=tensor.dtype,
        )

    def _work(self, tensor: Tensor) -> Tensor:
        return self.round(tensor.to(torch.float32))

    def add(self, lhs: Tensor, rhs: Tensor | float) -> Tensor:
        rhs = torch.as_tensor(rhs, device=lhs.device)
        return self._work(self._work(lhs) + self._work(rhs))

    def multiply(self, lhs: Tensor, rhs: Tensor | float) -> Tensor:
        rhs = torch.as_tensor(rhs, device=lhs.device)
        return self._work(self._work(lhs) * self._work(rhs))

    def residual_add(self, residual: Tensor, update: Tensor) -> Tensor:
        output_dtype = residual.dtype
        return self.add(residual, update).to(output_dtype)

    def rms_norm(
        self,
        hidden_states: Tensor,
        weight: Tensor | None,
        eps: float,
        *,
        weight_policy: "VectorRoundingPolicy | None" = None,
    ) -> Tensor:
        """Evaluate RMSNorm with rounding after each vector operation."""

        output_dtype = hidden_states.dtype
        x = self._work(hidden_states)
        square = self.multiply(x, x)
        mean_square = self._work(square.mean(dim=-1, keepdim=True))
        eps_tensor = torch.as_tensor(eps, device=x.device, dtype=torch.float32)
        denominator = self._work(torch.sqrt(self.add(mean_square, eps_tensor)))
        reciprocal = self._work(torch.reciprocal(denominator))
        normalized = self.multiply(x, reciprocal)
        if weight is not None:
            policy = weight_policy or self
            rounded_weight = (
                policy._work(weight) if policy.enabled else weight.to(torch.float32)
            )
            normalized = self._work(self._work(normalized) * rounded_weight)
        return normalized.to(output_dtype)

    def rope(
        self,
        query: Tensor,
        key: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        unsqueeze_dim: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate RoPE with rounded multiply and add boundaries."""

        query_dtype, key_dtype = query.dtype, key.dtype
        q, k = self._work(query), self._work(key)
        cos = self._work(cos).unsqueeze(unsqueeze_dim)
        sin = self._work(sin).unsqueeze(unsqueeze_dim)
        sequence_length = q.shape[-2]
        cos = cos[..., :sequence_length, :]
        sin = sin[..., :sequence_length, :]

        def rotate_half(tensor: Tensor) -> Tensor:
            half = tensor.shape[-1] // 2
            return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)

        query_out = self.add(
            self.multiply(q, cos),
            self.multiply(rotate_half(q), sin),
        )
        key_out = self.add(
            self.multiply(k, cos),
            self.multiply(rotate_half(k), sin),
        )
        return query_out.to(query_dtype), key_out.to(key_dtype)

    def softmax(self, tensor: Tensor, dim: int = -1) -> Tensor:
        """Evaluate stable softmax with rounded vector boundaries."""

        output_dtype = tensor.dtype
        x = self._work(tensor)
        maximum = self._work(x.amax(dim=dim, keepdim=True))
        shifted = self.add(x, -maximum)
        exponent = self._work(torch.exp(shifted))
        denominator = self._work(exponent.sum(dim=dim, keepdim=True))
        reciprocal = self._work(torch.reciprocal(denominator))
        return self.multiply(exponent, reciprocal).to(output_dtype)

    def silu(self, tensor: Tensor) -> Tensor:
        """Evaluate SiLU using rounded exp, reciprocal and multiply stages."""

        output_dtype = tensor.dtype
        x = self._work(tensor)
        negative = self._work(-x)
        exponent = self._work(torch.exp(torch.clamp(negative, -88.0, 88.0)))
        denominator = self.add(exponent, 1.0)
        sigmoid = self._work(torch.reciprocal(denominator))
        return self.multiply(x, sigmoid).to(output_dtype)

    def silu_gate(self, activation: Tensor, gate: Tensor) -> Tensor:
        """Round both SiLU and the following gated-product boundary."""

        output_dtype = activation.dtype
        silu = self.silu(activation)
        return self.multiply(silu, gate).to(output_dtype)


def round_vector_stage(tensor: Tensor, config: dict) -> Tensor:
    """Round a tensor with the vector policy encoded by ``config``."""

    return VectorRoundingPolicy.from_config(config).round(tensor)
