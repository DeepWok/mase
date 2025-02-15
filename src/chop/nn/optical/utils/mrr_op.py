"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-07-18 00:01:34
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-07-18 00:01:36
"""

from .compute import (
    complex_mult,
    polar_to_complex,
    polynomial,
)
import logging

import numpy as np
import torch

torch._C._jit_set_profiling_executor(False)


__all__ = [
    # "mrr_voltage_to_delta_lambda",
    # "mrr_tr_to_roundtrip_phase",
    # "mrr_roundtrip_phase_to_tr",
    "mrr_roundtrip_phase_to_tr_fused",
    # "mrr_roundtrip_phase_to_tr_grad_fused",
    "mrr_roundtrip_phase_to_tr_func",
    # "mrr_roundtrip_phase_to_out_phase",
    # "mrr_tr_to_out_phase",
    # "mrr_roundtrip_phase_to_tr_phase",
    # "mrr_roundtrip_phase_to_tr_phase_fused",
    # "mrr_modulator",
    # "mrr_filter",
    # "morr_filter",
    # "mrr_fwhm_to_ng",
    # "mrr_ng_to_fsr",
    # "mrr_finesse",
]


@torch.jit.script
def mrr_roundtrip_phase_to_tr_fused(
    rt_phi, a: float = 0.8, r: float = 0.9, intensity: bool = False
):
    """
    description:  round trip phase shift to field transmission
    rt_phi {torch.Tensor or np.ndarray} abs of roundtrip phase shift (abs(phase lag)). range from abs([-pi, 0])=[0, pi]\\
    a {scalar} attenuation coefficient\\
    r {scalar} self-coupling coefficient\\
    intensity {bool scalar} whether output intensity tranmission or field transmission\\
    return t {torch.Tensor or np.ndarray} mrr through port field/intensity transmission
    """

    # use slow but accurate mode from theoretical equation
    # create e^(-j phi) first

    # angle = -rt_phi
    # ephi = torch.view_as_complex(torch.stack([angle.cos(), angle.sin()], dim=-1)) ## this sign is from the negativity of phase lag
    # a_ephi = -a * ephi
    # t = torch.view_as_real((r + a_ephi).div(1 + r * a_ephi))
    # if(intensity):
    #     t = get_complex_energy(t)
    # else:
    #     t = get_complex_magnitude(t)
    ra_cosphi_by_n2 = -2 * r * a * rt_phi.cos()
    t = (a * a + r * r + ra_cosphi_by_n2) / (1 + r * r * a * a + ra_cosphi_by_n2)
    if not intensity:
        # as long as a is not equal to r, t cannot be 0.
        t = t.sqrt()

    return t


def mrr_roundtrip_phase_to_tr_func(
    a: float = 0.8, r: float = 0.9, intensity: bool = False
):
    c1 = -2 * a * r
    c2 = a * a + r * r
    c3 = 1 + r * r * a * a - a * a - r * r
    c4 = (a ** 2 - 1) * (r ** 2 - 1) * 2 * a * r

    class MRRRoundTripPhaseToTrFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            # ra_cosphi_by_n2 = input.cos().mul_(c1)
            # numerator = ra_cosphi_by_n2.add_(c2)
            # denominator = numerator.add(c3)
            # t = numerator / denominator
            t = input.cos().mul_(c1).add_(c2 + c3).reciprocal_().mul_(-c3).add_(1)
            if not intensity:
                # as long as a is not equal to r, t cannot be 0.
                t.sqrt_()
            return t

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            denominator = input.cos().mul_(c1).add_(c2 + c3)

            if intensity:
                denominator.square_()
                numerator = input.sin().mul_(c4)
            else:
                numerator = input.sin().mul_(c4 / 2)
                denominator = (
                    denominator.sub(1).pow_(1.5).mul_(denominator.sub(c3).sqrt_())
                )
            grad_input = numerator.div_(denominator).mul_(grad_output)
            return grad_input

    return MRRRoundTripPhaseToTrFunction.apply
