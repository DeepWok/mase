from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from chop.nn.quantizers import (
    integer_quantizer,
)


class _BatchNorm1dBase(torch.nn.BatchNorm1d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        self.bypass = False

        # NOTE(jlsand): In Torch, the batch norm classes refer to the learnable parameters gamma and beta
        # as weight and bias.
        self.x_quantizer = None
        self.w_quantizer = None
        self.b_quantizer = None

        # Register mean as a named parameter
        self.mean = torch.nn.Parameter(torch.zeros_like(self.weight))

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            # If bypass, there is no quantization. Let Torch handle the forward pass.
            return super().forward(x)
        else:
            # Could perhaps be done in a cleaner manner using a super() call.
            self._check_input_dim(x)

            # exponential_average_factor is set to self.momentum
            # (when it is available) only so that it gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:  # type: ignore[has-type]
                    self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(
                            self.num_batches_tracked
                        )
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            r"""
            Decide whether the mini-batch stats should be used for normalization rather than the buffers.
            Mini-batch stats are used in training mode, and in eval mode when buffers are None.
            """
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)

            # Quantise relevant parameters.
            x = self.x_quantizer(x)
            w = self.w_quantizer(self.weight)
            b = self.b_quantizer(self.bias)

            mean = self.w_quantizer(self.running_mean)
            var = self.b_quantizer(self.running_var)

            r"""
            Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
            passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
            used for normalization (i.e. in eval mode when buffers are not None).
            """
            return F.batch_norm(
                x,
                # If buffers are not to be tracked, ensure that they won't be updated
                mean if not self.training or self.track_running_stats else None,
                var if not self.training or self.track_running_stats else None,
                w,
                b,
                bn_training,
                exponential_average_factor,
                self.eps,
            )


class BatchNorm1dInteger(_BatchNorm1dBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        assert (
            config is not None
        ), "Attempted to initialise BatchNorm1dInteger with config as None"

        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]
        self.w_quantizer = partial(
            integer_quantizer, width=w_width, frac_width=w_frac_width
        )
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
        self.b_quantizer = partial(
            integer_quantizer, width=b_width, frac_width=b_frac_width
        )
