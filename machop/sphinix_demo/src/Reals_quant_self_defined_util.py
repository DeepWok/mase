# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import inspect
import math
from absl import logging
from torch import nn
import torch
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvTransposeNd
from pytorch_quantization.tensor_quant import (
    QuantDescriptor,
    QUANT_DESC_8BIT_PER_TENSOR,
    tensor_quant,
    fake_tensor_quant,
    TensorQuantFunction,
)
from pytorch_quantization.nn.modules.clip import Clip
from pytorch_quantization import calib
import pytorch_quantization.utils as quant_utils

class TensorQuantizer_TrueQuant(nn.Module):
    """Tensor quantizer module, Modifed for real quantization from pytorch_quantization package

    This module uses tensor_quant or fake_tensor_quant function to quantize a tensor. And wrappers variable, moving
    statistics we'd want when training a quantized network.

    Experimental features:
        ``clip`` stage learns range before enabling quantization.
        ``calib`` stage runs calibration

    Args:
        quant_desc: An instance of :func:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
        disabled: A boolean. If True, by pass the whole module returns input. Default False.
        if_quant: A boolean. If True, run main quantization body. Default True.
        if_clip: A boolean. If True, clip before quantization and learn amax. Default False.
        if_calib: A boolean. If True, run calibration. Not implemented yet. Settings of calibration will probably
            go to :func:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
    """

    use_fb_fake_quant = False

    def __init__(
        self,
        quant_desc=QuantDescriptor(),
        disabled=False,
        if_quant=True,
        if_clip=False,
        if_calib=False,
    ):
        """Initialize quantizer and set up required variables"""
        super(TensorQuantizer_TrueQuant, self).__init__()
        # Expand quant_desc. Use quant_desc.dict would be eaiser, but adding one-by-one explicitly gives more control
        self._num_bits = quant_desc.num_bits
        self._fake_quant = quant_desc.fake_quant
        self._axis = quant_desc.axis
        self._scale_amax = quant_desc.scale_amax
        self._learn_amax = quant_desc.learn_amax
        self._unsigned = quant_desc.unsigned
        self._narrow_range = quant_desc.narrow_range

        self._scale = 1
        self._disabled = disabled
        self._if_quant = if_quant
        self._if_clip = False
        self._if_calib = if_calib

        if quant_desc.amax is not None:
            self.register_buffer("_amax", torch.tensor(quant_desc.amax))

        # Clip module consumes a lot of memory, so only create it if learn_amax is True
        if self._learn_amax:
            init_amax = quant_desc.amax if quant_desc.amax is not None else 1.0
            self.clip = Clip(-init_amax, init_amax, learn_min=True, learn_max=True)
            # It makes more sense to enable clip stage (which learns amax) if learn_amax is true
            self.enable_clip()
        if if_clip:
            self.enable_clip()

        if quant_desc.calib_method == "histogram":
            logging.info("Creating histogram calibrator")
            self._calibrator = calib.HistogramCalibrator(
                num_bits=self._num_bits, axis=self._axis, unsigned=self._unsigned
            )
        elif quant_desc.calib_method == "max":
            logging.info("Creating Max calibrator")
            self._calibrator = calib.MaxCalibrator(
                num_bits=self._num_bits, axis=self._axis, unsigned=self._unsigned
            )

    # pylint:disable=missing-docstring
    @property
    def num_bits(self):
        return self._num_bits

    @property
    def unsigned(self):
        return self._unsigned

    @property
    def scale(self):
        if self._fake_quant:
            logging.error("Fake quantize mode doesn't use scale explicitly!")
        if self._scale is None:
            logging.critical("Accessing scale before quantizing any tensor!")
        return self._scale

    @property
    def amax(self):
        if not hasattr(self, "_amax"):
            return None
        return self._amax

    @property
    def step_size(self):
        if not hasattr(self, "_amax"):
            logging.error("step_size is undefined under dynamic amax mode!")
            return None
        return self._amax / (2.0 ** (self._num_bits - 1 + int(self._unsigned)) - 1.0)

    @property
    def axis(self):
        return self._axis

    @property
    def fake_quant(self):
        return self._fake_quant

    @property
    def narrow_range(self):
        return self._narrow_range

    def disable(self):
        """Bypass the module"""
        self._disabled = True

    def enable(self):
        self._disabled = False

    def disable_clip(self):
        """Disable clip stage"""
        self._if_clip = False
        self.clip.clip_value_min.requires_grad = False
        self.clip.clip_value_max.requires_grad = False

    def enable_clip(self):
        """Enable clip stage"""
        logging.warning("Enable `clip` stage for amax learning.")
        if not self._learn_amax:
            raise ValueError("learn_amax is False. Cannot enable clip.")
        self.clip.clip_value_min.requires_grad = True
        self.clip.clip_value_max.requires_grad = True
        self._if_clip = True

    def disable_calib(self):
        logging.warning("Disable {}".format(self._calibrator.__class__.__name__))
        self._if_calib = False

    def enable_calib(self):
        if self._calibrator is None:
            raise ValueError("Calibrator was not created, cannot enable calibration.")
        logging.info("Enable {}".format(self._calibrator.__class__.__name__))
        self._if_calib = True

    def disable_quant(self):
        logging.info("Disable `quant` stage.")
        self._if_quant = False

    def enable_quant(self):
        logging.info("Enable `quant` stage.")
        self._if_quant = True

    @amax.setter
    def amax(self, value):
        if value is None:
            logging.error("Setting amax no None is meaningless.")
        else:
            if isinstance(value, torch.Tensor):
                logging.warning("amax setter is not designed to take tensor.")
            if not hasattr(self, "_amax"):
                self.register_buffer("_amax", torch.tensor(value))
            else:
                value = torch.tensor(value, device=self._amax.device)
                if self._amax.shape != value.shape:
                    raise TypeError("Changing shape when setting amax is not allowed.")
                self._amax.data.copy_(value.data)

    @num_bits.setter
    def num_bits(self, value):
        self._num_bits = value

    @unsigned.setter
    def unsigned(self, value):
        self._unsigned = value

    @narrow_range.setter
    def narrow_range(self, value):
        self._narrow_range = value

    # pylint:enable=missing-docstring
    def load_calib_amax(self, *args, **kwargs):
        """Load amax from calibrator.

        Updates the amax buffer with value computed by the calibrator, creating it if necessary.
        *args and **kwargs are directly passed to compute_amax, except "strict" in kwargs. Refer to
        compute_amax for more details.
        """
        strict = kwargs.pop("strict", True)
        if getattr(self, "_calibrator", None) is None:
            raise RuntimeError("Calibrator not created.")
        calib_amax = self._calibrator.compute_amax(*args, **kwargs)
        if calib_amax is None:
            err_msg = "Calibrator returned None. This usually happens when calibrator hasn't seen any tensor."
            if not strict:
                logging.warning(err_msg)
                logging.warning("Set amax to NaN!")
                calib_amax = torch.tensor(math.nan)
            else:
                raise RuntimeError(
                    err_msg
                    + " Passing 'strict=False' to `load_calib_amax()` will ignore the error."
                )
        logging.warning("Load calibrated amax, shape={}.".format(calib_amax.shape))
        logging.log_first_n(
            logging.WARNING,
            "Call .cuda() if running on GPU after loading calibrated amax.",
            1,
        )
        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", calib_amax.data)
        else:
            self._amax.copy_(calib_amax)

    def init_learn_amax(self):
        """Initialize learned amax from fixed amax"""
        if self._learn_amax is False:
            raise RuntimeError("Called init_learn_amax with learn_amax=False.")
        logging.warning("Load amax as initial value for amax learning!")
        if self._amax.numel() != 1:
            logging.warning(
                "Per channel learned amax not supported. Initializing with max(amax)."
            )
            init_amax = torch.max(self._amax)
        else:
            init_amax = self._amax
        self.clip.clip_value_min.data.copy_(-init_amax.data)
        self.clip.clip_value_max.data.copy_(init_amax.data)

    def _get_amax(self, inputs):
        """get amax from buffer or compute it dynamically."""
        if hasattr(self, "_amax"):
            amax = self._amax
        else:
            if self._axis is None:
                reduce_axis = None
            else:
                reduce_axis = []
                # Swap axis to reduce
                axis = (
                    self._axis
                    if isinstance(self._axis, (list, tuple))
                    else [self._axis]
                )
                for i in range(inputs.dim()):
                    if not i in axis:
                        reduce_axis.append(i)
            amax = quant_utils.reduce_amax(
                inputs, axis=reduce_axis, keepdims=True
            ).detach()
        if self._scale_amax is not None:
            amax = amax.detach() * self._scale_amax

        return amax

    def _fb_fake_quant(self, inputs, amax):
        """Native pytorch fake quantization."""
        logging.log_first_n(
            logging.WARNING, "Use Pytorch's native experimental fake quantization.", 1
        )
        bound = (1 << (self._num_bits - 1 + int(self._unsigned))) - 1
        # To be consistent with ONNX, full range is used. e.g. range is [-128, 127] in int8
        if amax.numel() == 1:
            outputs = torch.fake_quantize_per_tensor_affine(
                inputs,
                amax.item() / bound,
                0,
                -bound - 1 if not self._unsigned else 0,
                bound,
            )
        else:
            amax_sequeeze = amax.squeeze().detach()
            if len(amax_sequeeze.shape) != 1:
                raise TypeError(
                    "Pytorch's native quantization doesn't support multiple axes"
                )
            quant_dim = list(amax.shape).index(list(amax_sequeeze.shape)[0])
            scale = amax_sequeeze / bound
            outputs = torch.fake_quantize_per_channel_affine(
                inputs,
                scale.data,
                torch.zeros_like(scale, dtype=torch.int32).data,
                quant_dim,
                -bound - 1 if not self._unsigned else 0,
                bound,
            )

        return outputs

    def _quant_forward(self, inputs):
        """Quantized forward pass."""
        if self._learn_amax:
            inputs = self.clip(inputs)
            amax = torch.max(
                -self.clip.clip_value_min, self.clip.clip_value_max
            ).detach()
        else:
            amax = self._get_amax(inputs)

        if self._fake_quant:
            if not TensorQuantizer_TrueQuant.use_fb_fake_quant:
                outputs = fake_tensor_quant(
                    inputs, amax, self._num_bits, self._unsigned, self._narrow_range
                )
            else:
                if inputs.dtype == torch.half or amax.dtype == torch.half:
                    raise Exception(
                        "Exporting to ONNX in fp16 is not supported. Please export in fp32, i.e. disable AMP."
                    )
                outputs = self._fb_fake_quant(inputs, amax)
        else:
            outputs, self._scale = TensorQuantFunction.apply(
                inputs, amax, self._num_bits, self._unsigned
            )
        return outputs

    def forward(self, inputs):
        """Apply tensor_quant function to inputs

        Args:
            inputs: A Tensor of type float32.

        Returns:
            outputs: A Tensor of type output_dtype
        """
        if self._disabled:
            return inputs

        outputs = inputs

        if self._if_calib:
            if self._calibrator is None:
                raise RuntimeError("Calibrator was not created.")
            # Shape is only know when it sees the first tensor
            self._calibrator.collect(inputs)

        if self._if_clip:
            if not self._learn_amax:
                raise RuntimeError("Clip without learning amax is not implemented.")
            outputs = self.clip(inputs)

        if self._if_quant:
            outputs = self._quant_forward(inputs)

        return outputs

    def _short_amax(self, fmt=".4f"):
        """Short description of amax

        Returns:
            'dynamic': if _amax is not registered
            'amax': if _amax is per-tensor
            '[min, max](size)': if _amax is per-channel
        """
        if not hasattr(self, "_amax"):
            return "dynamic"
        if self._amax.numel() == 1:
            return "{:{fmt}}".format(self._amax.item(), fmt=fmt)
        return "[{:{fmt}}, {:{fmt}}]({})".format(
            self._amax.min().item(),
            self._amax.max().item(),
            self._amax.numel(),
            fmt=fmt,
        )

    def extra_repr(self):
        if self._disabled:
            return "disabled"
        s = "{}{}bit".format("unsigned " if self._unsigned else "", self._num_bits)
        s += " narrow" if (self._narrow_range) else ""
        s += " fake" if (self._fake_quant) else ""
        s += " axis={}".format(self._axis) if self._axis is not None else " per-tensor"
        s += " amax={}".format(self._short_amax())
        s += " *{}".format(self._scale_amax) if self._scale_amax else ""
        s += " learned" if (self._learn_amax) else ""
        s += (
            " calibrator={}".format(self._calibrator.__class__.__name__)
            if (self._calibrator is not None)
            else ""
        )
        s += " scale={}".format(self._scale) if self._scale is not None else ""
        s += " quant" if (self._if_quant) else ""
        s += " clip" if (self._if_clip) else ""
        s += " calib" if (self._if_calib) else ""
        return s

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Overloaded module function

        Adds warnings during state_dict loading.
        A workaround is implemented for loading amax from checkpoint and only supports CUDA.

        Args:
            state_dict: A dict containing the state of the top level module
            prefix: A string that prefixes all of this modules state in state_dict, e.g. 'model.conv1.'
        """
        dst_has_amax = "_amax" in self._buffers
        src_has_amax = prefix + "_amax" in state_dict

        if not src_has_amax and dst_has_amax:
            logging.error("{}: No amax in state_dict.".format(prefix[:-1]))
        elif src_has_amax and not dst_has_amax:
            logging.debug(
                (
                    "{}: No '_amax' buffer to load amax into."
                    " '_amax` will be created as WAR for now. "
                    "This behavior will change in future."
                ).format(prefix[:-1])
            )
            self.register_buffer("_amax", state_dict[prefix + "_amax"].data.cuda())
        elif src_has_amax and dst_has_amax:
            logging.warning("{}: Overwriting amax.".format(prefix[:-1]))

        super(TensorQuantizer_TrueQuant, self)._load_from_state_dict(
            state_dict, prefix, *args, **kwargs
        )


class QuantMixin_TrueQuant:
    """
    Mixin class for adding basic quantization logic to quantized modules
    Modifed for real quantization from pytorch_quantization package.
    """

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_weight(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_weight = copy.deepcopy(value)

    def init_quantizer(self, quant_desc_input, quant_desc_weight, num_layers=None):
        """Helper function for __init__ of quantized module

        Create input and weight quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
            quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
            num_layers: An integer. Default None. If not None, create a list of quantizers.
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError(
                "{} should be only called by __init__ of quantized module.".format(
                    __name__
                )
            )
        logging.info(
            "Input is %squantized to %d bits in %s with axis %s!",
            "" if not quant_desc_input.fake_quant else "fake ",
            quant_desc_input.num_bits,
            self.__class__.__name__,
            quant_desc_input.axis,
        )
        logging.info(
            "Weight is %squantized to %d bits in %s with axis %s!",
            "" if not quant_desc_weight.fake_quant else "fake ",
            quant_desc_weight.num_bits,
            self.__class__.__name__,
            quant_desc_weight.axis,
        )

        if num_layers is None:
            self._input_quantizer = TensorQuantizer_TrueQuant(quant_desc_input)
            self._weight_quantizer = TensorQuantizer_TrueQuant(quant_desc_weight)
        else:
            self._input_quantizers = nn.ModuleList(
                [TensorQuantizer_TrueQuant(quant_desc_input) for _ in range(num_layers)]
            )
            self._weight_quantizers = nn.ModuleList(
                [
                    TensorQuantizer_TrueQuant(quant_desc_weight)
                    for _ in range(num_layers)
                ]
            )

    # pylint:disable=missing-docstring
    @property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def weight_quantizer(self):
        return self._weight_quantizer

    # pylint:enable=missing-docstring


class QuantInputMixin:
    """Mixin class for adding basic quantization logic to quantized modules"""

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    def init_quantizer(self, quant_desc_input):
        """Helper function for __init__ of simple quantized module

        Create input quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError(
                "{} should be only called by __init__ of quantized module.".format(
                    __name__
                )
            )
        self._fake_quant = True

        logging.info(
            "Input is %squantized to %d bits in %s with axis %s!",
            "" if not quant_desc_input.fake_quant else "fake ",
            quant_desc_input.num_bits,
            self.__class__.__name__,
            quant_desc_input.axis,
        )

        self._input_quantizer = TensorQuantizer_TrueQuant(quant_desc_input)

    # pylint:disable=missing-docstring
    @property
    def input_quantizer(self):
        return self._input_quantizer

    # pylint:enable=missing-docstring


def pop_quant_desc_in_kwargs(quant_cls, input_only=False, **kwargs):
    """Pop quant descriptors in kwargs

    If there is no descriptor in kwargs, the default one in quant_cls will be used

    Arguments:
       quant_cls: A class that has default quantization descriptors
       input_only: A boolean. If True, pop quant_desc_input only, not quant_desc_weight. Default false.

    Keyword Arguments:
       quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of input.
       quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of weight.
    """
    quant_desc_input = kwargs.pop(
        "quant_desc_input", quant_cls.default_quant_desc_input
    )
    if not input_only:
        quant_desc_weight = kwargs.pop(
            "quant_desc_weight", quant_cls.default_quant_desc_weight
        )

    # Check if anything is left in **kwargs
    if kwargs:
        raise TypeError("Unused keys: {}".format(kwargs.keys()))

    if input_only:
        return quant_desc_input
    return quant_desc_input, quant_desc_weight


#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Quantized Linear"""
from torch import nn
from torch.nn import functional as F

from pytorch_quantization import tensor_quant

__all__ = ["Linear", "QuantLinear"]


class _QuantConvNd_TrueQuant(torch.nn.modules.conv._ConvNd, QuantMixin_TrueQuant):
    """
    Base class of quantized Conv inherited from _ConvNd.
    Modified for real quantization from pytorch_quantization package.
    Comments of original arguments can be found in torch.nn.modules.conv.

    :param quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
        Quantization descriptor of input.
    :param quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
        Quantization descriptor of weight.

    :raises ValueError: If unsupported arguments are passed in.

    Readonly properties:
        - input_quantizer
        - weight_quantizer

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        quant_desc_input,
        quant_desc_weight,
    ):
        super(_QuantConvNd, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
        )
        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def _quant(self, input):
        """Apply quantization on input and weight

        Function called by the classes lower in the hierarchy, which actually performs the quantization before forward
        in the derivate class the particular Function.

        Arguments:
            input: in_features to quantize
        Returns:
            A tuple: (quant_in_feature, quant_weight)
        """
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        return (quant_input, quant_weight)


class QuantConv2d_TrueQuant(_QuantConvNd_TrueQuant):
    """Quantized 2D conv"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        **kwargs
    ):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        quant_desc_input, quant_desc_weight = pop_quant_desc_in_kwargs(
            self.__class__, **kwargs
        )
        super(QuantConv2d_TrueQuant, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            quant_desc_input=quant_desc_input,
            quant_desc_weight=quant_desc_weight,
        )

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input, quant_weight = self._quant(input)

        if self.padding_mode == "circular":
            expanded_padding = (
                (self.padding[1] + 1) // 2,
                self.padding[1] // 2,
                (self.padding[0] + 1) // 2,
                self.padding[0] // 2,
            )
            output = F.conv2d(
                F.pad(quant_input, expanded_padding, mode="circular"),
                quant_weight,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        else:
            output = F.conv2d(
                quant_input,
                quant_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        return output


class QuantLinear_TrueQuant(nn.Linear, QuantMixin_TrueQuant):
    """
    Quantized version of nn.Linear
    Modifed for real quantization from pytorch_quantization package.

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
        Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
        Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinear_TrueQuant, self).__init__(in_features, out_features, bias)
        quant_desc_input, quant_desc_weight = pop_quant_desc_in_kwargs(
            self.__class__, **kwargs
        )

        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)
        output = F.linear(quant_input, quant_weight, bias=self.bias)

        return output


class _QuantConvNd(torch.nn.modules.conv._ConvNd, QuantMixin_TrueQuant):
    """base class of quantized Conv inherited from _ConvNd

    Comments of original arguments can be found in torch.nn.modules.conv

    Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        quant_desc_input,
        quant_desc_weight,
    ):
        super(_QuantConvNd_TrueQuant, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
        )
        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def _quant(self, input):
        """Apply quantization on input and weight

        Function called by the classes lower in the hierarchy, which actually performs the quantization before forward
        in the derivate class the particular Function.

        Arguments:
            input: in_features to quantize
        Returns:
            A tuple: (quant_in_feature, quant_weight)
        """
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        return (quant_input, quant_weight)


Linear = QuantLinear_TrueQuant
