import torch
from chop.nn.quantizers import integer_floor_quantizer
from functools import partial
import torch.nn.functional as F
from torch import Tensor


def mxint_quantize(x, width: int = 12, exponent_width: int = 6, exponent: int = None):
    """
    - Convert IEEE FP32/64 to Microsoft floating point (MSFP), where an exponent is shared over all elements in a block.
    - `e_shared x [(-1)^s1 x mantissa1, (-1)^s2 x mantissa2, ...]`
    - See https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf

    ---
    - forward: convert IEEE FP32/64 to MSFP
    - backward: STE

    ---
    - `width`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_width`: the number of exponent bits, which is shared over a block
    - `exponent_bias`: the exponent bias, if None, `2**(exponent_bits-1)-1` will be used
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    exponent_bias = 2 ** (exponent_width - 1)

    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias

    # exponent
    if exponent == None:
        exponent = torch.ceil(torch.log2(x.abs().max())) - exponent_bias
        exponent = torch.clamp(exponent, exponent_min, exponent_max)
    # mantissa
    int_min = -(2 ** (width - 1))
    int_max = 2 ** (width - 1) - 1
    mantissa = x / 2**exponent
    mantissa = torch.clamp(mantissa.floor(), int_min, int_max)
    msfp_x = (2**exponent) * mantissa
    return msfp_x, mantissa, exponent


def block_mxint_quant(tensor, q_config, parallelism):
    original_shape = tensor.shape
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    if len(parallelism) == 1:
        parallelism = [1, parallelism[0]]

    p1 = parallelism[0]
    p0 = parallelism[1]
    t1 = tensor.shape[-2]
    t0 = tensor.shape[-1]
    reshaped_tensor = tensor.reshape(-1, t1 // p1, p1, t0 // p0, p0).permute(
        0, 1, 3, 2, 4
    )

    # Quantize
    quantizer = partial(mxint_quantize, **q_config)
    reshaped_tensor = torch.tensor(reshaped_tensor.reshape(-1, p1 * p0))
    mtensor = torch.zeros(reshaped_tensor.shape)
    etensor = torch.zeros(reshaped_tensor.shape[0])
    for i in range(reshaped_tensor.shape[0]):
        reshaped_tensor[i], mtensor[i], etensor[i] = quantizer(reshaped_tensor[i])
    qtensor = reshaped_tensor.reshape(-1, t1 // p1, t0 // p0, p1, p0).permute(
        0, 1, 3, 2, 4
    )
    mtensor = (
        mtensor.reshape(-1, t1 // p1, t0 // p0, p1, p0)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, t1, t0)
    )
    etensor = etensor.reshape(-1, t1 // p1, t0 // p0)
    return qtensor.reshape(original_shape), mtensor.reshape(original_shape), etensor


def pack_tensor_to_mx_listed_chunk(mtensor, etensor, parallelism):
    if len(mtensor.shape) == 1:
        mtensor = mtensor.unsqueeze(0)
    if len(parallelism) == 1:
        parallelism = [1, parallelism[0]]
    p1 = parallelism[0]
    p0 = parallelism[1]
    t1 = mtensor.shape[-2]
    t0 = mtensor.shape[-1]
    reshaped_mtensor = (
        mtensor.reshape(-1, t1 // p1, p1, t0 // p0, p0)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, p1 * p0)
    )
    reshaped_etensor = etensor.reshape(-1)
    mx_data_list = []
    for i in range(reshaped_mtensor.shape[0]):
        mx_data_list.append(
            (reshaped_mtensor[i].int().tolist(), int(reshaped_etensor[i]))
        )
    return mx_data_list


from chop.nn.quantized.modules.linear import _LinearBase


class MXIntLinear(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
        out_config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.out_config = out_config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizer
        w_width, w_exponent_width = (
            config["weight_width"],
            config["weight_exponent_width"],
        )
        w_p1, w_p0 = (
            config["weight_parallelism_dim_1"],
            config["weight_parallelism_dim_0"],
        )
        x_width, x_exponent_width = (
            config["data_in_width"],
            config["data_in_exponent_width"],
        )
        x_p1, x_p0 = (
            config["data_in_parallelism_dim_1"],
            config["data_in_parallelism_dim_0"],
        )
        # check bias quantizer, if not, use weight quantizer
        b_width, b_exponent_width = config["bias_width"], config["bias_exponent_width"]
        b_p1, b_p0 = config["bias_parallelism_dim_1"], config["bias_parallelism_dim_0"]
        base_quantizer = block_mxint_quant
        if out_config is not None:
            out_width, out_exponent_width = (
                config["data_out_width"],
                config["data_out_exponent_width"],
            )
            out_p1, out_p0 = (
                config["data_out_parallelism_dim_1"],
                config["data_out_parallelism_dim_0"],
            )
            self.out_quantizer = partial(
                base_quantizer,
                q_config={"width": out_width, "exponent_width": out_exponent_width},
                parallelism=[out_p1, out_p0],
            )
        self.w_quantizer = partial(
            base_quantizer,
            q_config={"width": w_width, "exponent_width": w_exponent_width},
            parallelism=[w_p1, w_p0],
        )
        self.x_quantizer = partial(
            base_quantizer,
            q_config={"width": x_width, "exponent_width": x_exponent_width},
            parallelism=[x_p1, x_p0],
        )
        self.b_quantizer = partial(
            base_quantizer,
            q_config={"width": b_width, "exponent_width": b_exponent_width},
            parallelism=[b_p1, b_p0],
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        else:
            x, mx, ex = self.x_quantizer(x)
            w, mw, ew = self.w_quantizer(self.weight)
            if self.bias is not None:
                bias, mb, eb = self.b_quantizer(self.bias)
            else:
                bias = None
            out = F.linear(x, w, bias)
            # print(f"mout = {F.linear(mx, mw, mb*2**(ex+ew - eb).floor())}")
            if self.out_quantizer is None:
                return out
            return self.out_quantizer(out)
