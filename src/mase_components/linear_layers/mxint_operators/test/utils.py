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
    mantissa = x * (2**(width - 1)) / 2**exponent
    # print(mantissa, int_min, int_max)
    mantissa = torch.clamp(mantissa.floor(), int_min, int_max)
    q_x = (2**exponent) * mantissa /((2**(width - 1)))
    return q_x, mantissa, exponent


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
            print((mx @ mw.transpose(0, 1)).int())
            if self.bias is not None:
                bias, mb, eb = self.b_quantizer(self.bias)
            else:
                bias = None
            breakpoint()
            out = F.linear(x, w, bias)
            # print(f"mout = {F.linear(mx, mw, mb*2**(ex+ew - eb).floor())}")
            if self.out_quantizer is None:
                return out
            return self.out_quantizer(out)

def MXIntMatmulHardware(man_x, exp_x, man_y, exp_y, x_config, y_config):
    """
    assume 2 dimensional input
    config = {
        "width": ,
        "exp_width" ,
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
    # assume 2 dimensional input
    assert x_config["depth_dim_0"] == y_config["depth_dim_1"], "need to check the setting of dim"
    assert x_config["depth_dim_0"] == y_config["depth_dim_1"], "need to check the setting of dim"
    def block_wise_reshape_tensor(x, exp_x, x_config):
        reshaped_x = x.reshape(x_config["depth_dim_1"],x_config["parallism_dim_1"], x_config["depth_dim_0"],x_config["parallism_dim_0"]).permute(0,1,3,2)
        reshaped_x = reshaped_x.reshape(x_config["depth_dim_1"] * x_config["depth_dim_0"], x_config["parallism_dim_1"], x_config["parallism_dim_0"])
        return reshaped_x
    reshaped_exp_x = exp_x.reshape(-1)
    reshaped_man_x = block_wise_reshape_tensor(man_x)
    reshaped_exp_y = exp_y.reshape(-1)
    reshaped_man_y = block_wise_reshape_tensor(man_y)
    for i in range(x_config["depth_dim_1"]):
        for j in range(y_config["depth_dim_0"]):
            partial_man_out = torch.zeros(y_config["depth_dim_1"], x_config["parallism_dim_1"], y_config["parallism_dim_0"])
            partial_exp_out = torch.zeros(y_config["depth_dim_1"])
            for k in range(y_config["depth_dim_1"]):
                man_x_block = reshaped_man_x[i*x_config["depth_dim_1"] + k]
                exp_x_block = reshaped_exp_x[i*x_config["depth_dim_1"] + k]
                man_y_block = reshaped_man_y[k*y_config["depth_dim_0"] + j]
                exp_y_block = reshaped_exp_y[k*y_config["depth_dim_0"] + j]
                partial_man_out[k], partial_exp_out[k] = MatmulCore(man_x_block, exp_x_block, man_y_block,exp_y_block)
            acc_man_out, acc_exp_out = MxIntAccumulator(partial_man_out.reshape(y_config["depth_dim_1"], -1), partial_exp_out)
                
    # config 

def MxIntCast(man_in, exp_in, param):
    # In Man Width
    max_in = torch.ceil(torch.log2(man_in.abs().max()))
    out_width = param["out_width"]
    out_exponent_width = param["out_exponent_width"]
    in_width = param["in_width"]
    in_frac_width = param["in_frac_width"]
    in_exponent_width = param["in_exponent_width"]

    out_exponent_max = 2**(out_exponent_width - 1) - 1
    out_exponent_min = -2**(out_exponent_width - 1)
    
    out_min = -(2 ** (out_width - 1))
    out_max = 2 ** (out_width - 1) - 1
    lma_in = torch.ceil(torch.log2(man_in.abs().max() + 1e-3))
    out_exp_full = lma_in + exp_in -  in_frac_width
    out_exp = torch.clamp(out_exp_full, out_exponent_min, out_exponent_max)
    out_man = man_in / 2**(in_frac_width - exp_in + out_exp - (out_width - 1))
    out_man = torch.clamp(out_man, out_min, out_max)

    return out_man, out_exp
    
    
    

def MatmulCore(man_x, exp_x, man_y, exp_y):
    return man_x @ man_y, exp_x + exp_y
    
def MxIntAccumulator(man, exp):
    IN_DEPTH, BLOCK_SIZE = man.shape[0],man.shape[1]
    max_exp = torch.Tensor([-999])
    mout = torch.zeros(BLOCK_SIZE)
    out_exp = torch.Tensor([-999])
    for i in range(IN_DEPTH):
        max_exp = exp[i] if exp[i]>max_exp else max_exp
        mout = mout // 2**(max_exp - out_exp)
        out_exp = max_exp
        shifted_man = man[i] // 2**(max_exp - exp[i])
        mout = mout + shifted_man
    
    return mout, out_exp
