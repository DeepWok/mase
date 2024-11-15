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
        exponent = torch.ceil(torch.log2(x.abs().max()))
        exponent = torch.clamp(exponent, exponent_min, exponent_max)
    # mantissa
    int_min = -(2 ** (width - 1))
    int_max = 2 ** (width - 1) - 1
    mantissa = x * (2 ** (width - 1)) / 2**exponent
    # print(mantissa, int_min, int_max)
    mantissa = torch.clamp(mantissa.floor(), int_min, int_max)
    q_x = (2**exponent) * mantissa / ((2 ** (width - 1)))
    return q_x, mantissa, exponent


def block_mxint_quant(tensor, q_config, parallelism):
    print(tensor)
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


class MXIntLinearHardware(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizer
        w_width, w_exponent_width = (
            config["weight_width"],
            config["weight_exponent_width"],
        )
        w_p1, w_p0 = (
            config["weight_parallelism"][0],
            config["weight_parallelism"][1],
        )
        x_width, x_exponent_width = (
            config["data_in_width"],
            config["data_in_exponent_width"],
        )
        x_p1, x_p0 = (
            config["data_in_parallelism"][0],
            config["data_in_parallelism"][1],
        )
        # check bias quantizer, if not, use weight quantizer
        b_width, b_exponent_width = config["bias_width"], config["bias_exponent_width"]
        b_p1, b_p0 = (
            config["bias_parallelism"][0],
            config["bias_parallelism"][1],
        )
        base_quantizer = block_mxint_quant
        out_width, out_exponent_width = (
            config["data_out_width"],
            config["data_out_exponent_width"],
        )
        out_p1, out_p0 = (
            config["data_out_parallelism"][0],
            config["data_out_parallelism"][1],
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
        print(x)
        x, mx, ex = self.x_quantizer(x)
        in_x = (mx, ex)
        w, mw, ew = self.w_quantizer(self.weight)
        in_w = (mw, ew)
        if self.bias is not None:
            bias, mbias, ebias = self.b_quantizer(self.bias)
            in_bias = (mbias, ebias)
        else:
            bias = None
            in_bias = None

        out = wrapped_mxint_linear_hardware(
            in_x, in_w, in_bias, self.in_features, self.out_features, self.config
        )

        return out


def wrapped_mxint_linear_hardware(x, w, bias, in_features, out_features, config):
    mx = x[0]
    n = mx.reshape(-1, in_features).shape[0]
    in_config = {
        "x_config": {
            "width": config["data_in_width"],
            "exponent_width": config["data_in_exponent_width"],
            "parallism_dim_0": config["data_in_parallelism"][1],
            "parallism_dim_1": config["data_in_parallelism"][0],
            "depth_dim_0": in_features // config["data_in_parallelism"][1],
            "depth_dim_1": n // config["data_in_parallelism"][0],
            "dim_0": in_features,
            "dim_1": n,
        },
        "w_config": {
            "width": config["weight_width"],
            "exponent_width": config["weight_exponent_width"],
            "parallism_dim_0": config["weight_parallelism"][1],
            "parallism_dim_1": config["weight_parallelism"][0],
            "depth_dim_0": in_features // config["weight_parallelism"][1],
            "depth_dim_1": out_features // config["weight_parallelism"][0],
            "dim_0": in_features,
            "dim_1": out_features,
        },
        "bias_config": {
            "width": config["bias_width"],
            "exponent_width": config["bias_exponent_width"],
            "parallism_dim_0": config["bias_parallelism"][1],
            "parallism_dim_1": 1,
            "depth_dim_0": out_features // config["bias_parallelism"][1],
            "depth_dim_1": 1,
            "dim_0": out_features,
            "dim_1": 1,
        },
        "out_config": {
            "width": config["data_out_width"],
            "exponent_width": config["data_out_exponent_width"],
            "parallism_dim_0": config["data_out_parallelism"][1],
            "parallism_dim_1": config["data_out_parallelism"][0],
            "depth_dim_0": out_features // config["data_out_parallelism"][1],
            "depth_dim_1": n // config["data_out_parallelism"][0],
            "dim_0": out_features,
            "dim_1": n,
        },
    }
    mout, eout = mxint_linear_hardware(x, w, bias, in_config)
    out_config = in_config["out_config"]
    reshaped_mout = mout.reshape(
        out_config["depth_dim_1"],
        out_config["parallism_dim_1"],
        out_config["depth_dim_0"],
        out_config["parallism_dim_0"],
    ).permute(0, 2, 1, 3)
    reshaped_out = reshaped_mout * 2 ** (
        eout[:, :, None, None] - config["data_out_width"] + 1
    )
    out = reshaped_out.reshape(
        out_config["depth_dim_1"],
        out_config["depth_dim_0"],
        out_config["parallism_dim_1"],
        out_config["parallism_dim_0"],
    ).permute(0, 2, 1, 3)
    out = out.reshape(out_config["dim_1"], out_config["dim_0"])

    return out


def mxint_linear_hardware(x, w, bias, config):
    """
    assume 2 dimensional input
    config = {
        "x_config":{
            "width": ,
            "exponent_width" ,
            "parallism_dim_0",
            "parallism_dim_1",
            "depth_dim_0",
            "depth_dim_1",
            "dim_0",
            "dim_1",
        },
        "w_config": {
            ...
        },
        "bias_config": {
            ...
        },
        "out_config": {
            ...
        },
    }
    """
    mx, ex = x
    mw, ew = w
    x_config = config["x_config"]
    w_config = config["w_config"]
    out_config = config["out_config"]
    from math import ceil, log2

    def DotProductCore(man_x, exp_x, man_y, exp_y):
        return man_x @ man_y.transpose(0, 1), exp_x + exp_y

    def block_wise_reshape_tensor(x, x_config):
        reshaped_x = x.reshape(
            x_config["depth_dim_1"],
            x_config["parallism_dim_1"],
            x_config["depth_dim_0"],
            x_config["parallism_dim_0"],
        ).permute(0, 2, 1, 3)
        reshaped_x = reshaped_x.reshape(
            x_config["depth_dim_1"] * x_config["depth_dim_0"],
            x_config["parallism_dim_1"],
            x_config["parallism_dim_0"],
        )
        return reshaped_x

    # assume 2 dimensional input
    assert (
        x_config["depth_dim_0"] == w_config["depth_dim_0"]
    ), "need to check the setting of dim"
    assert (
        x_config["parallism_dim_0"] == w_config["parallism_dim_0"]
    ), "need to check the setting of dim"
    reshaped_ex = ex.reshape(-1)
    reshaped_mx = block_wise_reshape_tensor(mx, x_config)
    reshaped_ew = ew.reshape(-1)
    reshaped_mw = block_wise_reshape_tensor(mw, w_config)
    man_out = torch.zeros(
        x_config["depth_dim_1"],
        w_config["depth_dim_1"],
        x_config["parallism_dim_1"] * w_config["parallism_dim_1"],
    )
    exp_out = torch.zeros(x_config["depth_dim_1"], w_config["depth_dim_1"])
    for i in range(x_config["depth_dim_1"]):
        for j in range(w_config["depth_dim_1"]):
            partial_man_out = torch.zeros(
                w_config["depth_dim_0"],
                x_config["parallism_dim_1"],
                w_config["parallism_dim_1"],
            )
            partial_exp_out = torch.zeros(w_config["depth_dim_0"])
            for k in range(x_config["depth_dim_0"]):
                mx_block = reshaped_mx[i * x_config["depth_dim_0"] + k]
                ex_block = reshaped_ex[i * x_config["depth_dim_0"] + k]
                mw_block = reshaped_mw[j * w_config["depth_dim_0"] + k]
                ew_block = reshaped_ew[j * w_config["depth_dim_0"] + k]
                partial_man_out[k], partial_exp_out[k] = DotProductCore(
                    mx_block, ex_block, mw_block, ew_block
                )
            acc_man_out, acc_exp_out = MxIntAccumulator(
                partial_man_out.reshape(w_config["depth_dim_0"], -1), partial_exp_out
            )
            if bias != None:
                bias_config = config["bias_config"]
                mbias, ebias = bias
                reshaped_mbias = mbias.reshape(
                    w_config["depth_dim_1"], w_config["parallism_dim_1"]
                )
                reshaped_ebias = ebias.reshape(w_config["depth_dim_1"])
                shifted_value = (
                    reshaped_ebias[j]
                    - acc_exp_out
                    + x_config["width"]
                    + w_config["width"]
                    - 2
                    - (bias_config["width"] - 1)
                )
                shifted_bias = reshaped_mbias[j].repeat(
                    x_config["parallism_dim_1"]
                ) * 2 ** (shifted_value)
                print(reshaped_mbias[j])
                print(shifted_value)
                acc_man_out = shifted_bias + acc_man_out
                print("shfited_bias", shifted_bias)
            man_out[i][j], exp_out[i][j] = MxIntCast(
                acc_man_out,
                acc_exp_out,
                {
                    "in_width": x_config["width"]
                    + w_config["width"]
                    + ceil(log2(x_config["dim_0"])),
                    "in_frac_width": x_config["width"] + w_config["width"] - 2,
                    "in_exponent_width": max(
                        x_config["exponent_width"], w_config["exponent_width"]
                    )
                    + 1,
                    "out_width": out_config["width"],
                    "out_exponent_width": out_config["exponent_width"],
                },
            )
    man_out = (
        man_out.reshape(
            x_config["depth_dim_1"],
            w_config["depth_dim_1"],
            x_config["parallism_dim_1"],
            w_config["parallism_dim_1"],
        )
        .permute(0, 2, 1, 3)
        .reshape(x_config["dim_1"], w_config["dim_1"])
    )
    return man_out, exp_out


def MXIntMatmulHardware(man_x, exp_x, man_y, exp_y, x_config, y_config, out_config):
    """
    assume 2 dimensional input
    config = {
        "width": ,
        "exponent_width" ,
        "parallism_dim_0",
        "parallism_dim_1",
        "depth_dim_0",
        "depth_dim_1",
        "dim_0",
        "dim_1",
    }
    man.shape = [dim_1 * dim_0]
    exp.shape = [depth_dim_1,  depth_dim_0]
    """
    from math import ceil, log2

    def MatmulCore(man_x, exp_x, man_y, exp_y):
        return man_x @ man_y, exp_x + exp_y

    # assume 2 dimensional input
    assert (
        x_config["depth_dim_0"] == y_config["depth_dim_1"]
    ), "need to check the setting of dim"

    def block_wise_reshape_tensor(x, x_config):
        reshaped_x = x.reshape(
            x_config["depth_dim_1"],
            x_config["parallism_dim_1"],
            x_config["depth_dim_0"],
            x_config["parallism_dim_0"],
        ).permute(0, 2, 1, 3)
        reshaped_x = reshaped_x.reshape(
            x_config["depth_dim_1"] * x_config["depth_dim_0"],
            x_config["parallism_dim_1"],
            x_config["parallism_dim_0"],
        )
        return reshaped_x

    reshaped_exp_x = exp_x.reshape(-1)
    reshaped_man_x = block_wise_reshape_tensor(man_x, x_config)
    reshaped_exp_y = exp_y.reshape(-1)
    reshaped_man_y = block_wise_reshape_tensor(man_y, y_config)
    man_out = torch.zeros(
        x_config["depth_dim_1"],
        y_config["depth_dim_0"],
        x_config["parallism_dim_1"] * y_config["parallism_dim_0"],
    )
    exp_out = torch.zeros(x_config["depth_dim_1"], y_config["depth_dim_0"])
    for i in range(x_config["depth_dim_1"]):
        for j in range(y_config["depth_dim_0"]):
            partial_man_out = torch.zeros(
                y_config["depth_dim_1"],
                x_config["parallism_dim_1"],
                y_config["parallism_dim_0"],
            )
            partial_exp_out = torch.zeros(y_config["depth_dim_1"])
            for k in range(y_config["depth_dim_1"]):
                man_x_block = reshaped_man_x[i * x_config["depth_dim_0"] + k]
                exp_x_block = reshaped_exp_x[i * x_config["depth_dim_0"] + k]
                man_y_block = reshaped_man_y[k * y_config["depth_dim_0"] + j]
                exp_y_block = reshaped_exp_y[k * y_config["depth_dim_0"] + j]
                partial_man_out[k], partial_exp_out[k] = MatmulCore(
                    man_x_block, exp_x_block, man_y_block, exp_y_block
                )
            acc_man_out, acc_exp_out = MxIntAccumulator(
                partial_man_out.reshape(y_config["depth_dim_1"], -1), partial_exp_out
            )
            man_out[i][j], exp_out[i][j] = MxIntCast(
                acc_man_out,
                acc_exp_out,
                {
                    "in_width": x_config["width"]
                    + y_config["width"]
                    + ceil(log2(x_config["dim_0"])),
                    "in_frac_width": x_config["width"] + y_config["width"] - 2,
                    "in_exponent_width": max(
                        x_config["exponent_width"], y_config["exponent_width"]
                    )
                    + 1,
                    "out_width": out_config["width"],
                    "out_exponent_width": out_config["exponent_width"],
                },
            )
    man_out = (
        man_out.reshape(
            x_config["depth_dim_1"],
            y_config["depth_dim_0"],
            x_config["parallism_dim_1"],
            x_config["parallism_dim_0"],
        )
        .permute(0, 2, 1, 3)
        .reshape(x_config["dim_1"], y_config["dim_0"])
    )
    return man_out, exp_out


def MxIntCast(man_in, exp_in, param):
    # In Man Width
    max_in = torch.ceil(torch.log2(man_in.abs().max()))
    out_width = param["out_width"]
    out_exponent_width = param["out_exponent_width"]
    in_width = param["in_width"]
    in_frac_width = param["in_frac_width"]
    in_exponent_width = param["in_exponent_width"]

    out_exponent_max = 2 ** (out_exponent_width - 1) - 1
    out_exponent_min = -(2 ** (out_exponent_width - 1))

    out_min = -(2 ** (out_width - 1))
    out_max = 2 ** (out_width - 1) - 1
    lma_in = torch.ceil(torch.log2(man_in.abs().max() + 1e-3))
    out_exp_full = lma_in + exp_in - in_frac_width
    out_exp = torch.clamp(out_exp_full, out_exponent_min, out_exponent_max)
    out_man = man_in // 2 ** (in_frac_width - exp_in + out_exp - (out_width - 1))
    out_man = torch.clamp(out_man, out_min, out_max)

    return out_man, out_exp


# def MxIntAccumulator(man, exp, clamp_width = 15):
#     IN_DEPTH, BLOCK_SIZE = man.shape[0],man.shape[1]
#     min_exp = torch.Tensor([64])
#     mout = torch.zeros(BLOCK_SIZE)
#     out_exp = torch.Tensor([64])
#     for i in range(IN_DEPTH):
#         min_exp = exp[i] if exp[i]<min_exp else min_exp
#         mout = mout * 2**(out_exp - min_exp)
#         mout = torch.clamp(mout, -2 ** (clamp_width - 1), 2 ** (clamp_width -1) - 1)
#         out_exp = min_exp
#         shifted_man = man[i] * 2**(exp[i]-min_exp)
#         shifted_man = torch.clamp(shifted_man, -2 ** (clamp_width - 1), 2 ** (clamp_width -1) - 1)
#         mout = mout + shifted_man

#     return mout, out_exp


def MxIntAccumulator(man, exp):
    IN_DEPTH, BLOCK_SIZE = man.shape[0], man.shape[1]
    max_exp = torch.Tensor([-999])
    mout = torch.zeros(BLOCK_SIZE)
    out_exp = torch.Tensor([-999])
    for i in range(IN_DEPTH):
        max_exp = exp[i] if exp[i] > max_exp else max_exp
        mout = mout // 2 ** (max_exp - out_exp)
        out_exp = max_exp
        shifted_man = man[i] // 2 ** (max_exp - exp[i])
        mout = mout + shifted_man

    return mout, out_exp
