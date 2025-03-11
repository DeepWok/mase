from chop.nn.quantized.modules.linear import _LinearBase
from torch import Tensor
import torch
from .quantizers import mxint_hardware, reshape_to_block, reshape_back

def fast_linear(x, w, b, config):
    batch_size, n = x.shape[:2]
    out_features = w.shape[0]
    qx, mx, ex = mxint_hardware(x, **{
        "parallelism":[config["x_config"]["parallism_dim_1"], config["x_config"]["parallism_dim_0"]],
        "q_config":{
            "width": config["x_config"]["width"],
            "exponent_width": config["x_config"]["exponent_width"],
            "round_bits": config["round_bits"],
            
        },
    })
    qw, mw, ew = mxint_hardware(w, **{
        "parallelism":[config["w_config"]["parallism_dim_1"], config["w_config"]["parallism_dim_0"]],
        "q_config":{
            "width": config["w_config"]["width"],
            "exponent_width": config["w_config"]["exponent_width"],
            "round_bits": 8,
        }
    })
    qb, mb, eb = mxint_hardware(b, **{
        "parallelism":[config["bias_config"]["parallism_dim_1"], config["bias_config"]["parallism_dim_0"]],
        "q_config":{
            "width": config["bias_config"]["width"],
            "exponent_width": config["bias_config"]["exponent_width"],
            "round_bits": 8,
        }
    })
    x_config = config["x_config"]
    w_config = config["w_config"]
    reshaped_mx = reshape_to_block(mx, x_config["dim_1"], x_config["dim_0"], x_config["parallism_dim_1"], x_config["parallism_dim_0"])
    reshaped_mw = reshape_to_block(mw, w_config["dim_1"], w_config["dim_0"], w_config["parallism_dim_1"], w_config["parallism_dim_0"])

    # move the infeatures depth to the front
    mx_for_accumulation = reshaped_mx.permute(2, 0, 1, 3, 4)
    # The dimension will be [depth_in_features, batch_size, depth_n, parallism_n, parallism_in_features]
    # For every parallelised block, we will have a unique exponent
    # Original shape of ex is [batch_size, depth_n, depth_in_features]
    # We will permute it to [depth_in_features, batch_size, depth_n]
    ex_for_accumulation = ex.permute(2, 0, 1)

    # Same for mw, the shape of mw is [depth_out_features, depth_in_features, parallism_out_features, parallism_in_features]
    mw_for_accumulation = reshaped_mw.squeeze(0)
    mw_for_accumulation = mw_for_accumulation.permute(1, 0, 2, 3)
    ew_for_accumulation = ew.transpose(0, 1)

    # We are trying to do the matmul based on the block partition
    # mx is [depth_in_features, batch_size, depth_n, parallism_n, parallism_in_features]
    # mw is [depth_in_features, depth_out_features, parallism_out_features, parallism_in_features]
    # merge depth_out_features and parallelism_out_features
    # mw = [depth_in_features, out_features, parallism_in_features]
    mw_for_accumulation = mw_for_accumulation.reshape(mw_for_accumulation.shape[0], -1, mw_for_accumulation.shape[-1])

    mout = mx_for_accumulation[0] @ mw_for_accumulation[0].transpose(-2, -1)
    mout = reshape_to_block(mout, x_config["dim_1"], w_config["dim_1"], x_config["parallism_dim_1"], w_config["parallism_dim_1"])
    # shape of mout is [batch_size, depth_n, parallism_n, out_features]
    ex_expanded = ex_for_accumulation.unsqueeze(-1)  # [depth_in_features,  batch_size,     depth_n,    1]
    ew_expanded = ew_for_accumulation.unsqueeze(1).unsqueeze(2)  # [depth_in_features,   1,            1,          depth_out_features]
    eout = (ex_expanded[0] + ew_expanded[0]).unsqueeze(-1).unsqueeze(-1)
    for i in range(1, mx_for_accumulation.shape[0]):
        new_exponent = (ex_expanded[i] + ew_expanded[i]).unsqueeze(-1).unsqueeze(-1)
        max_exponent = torch.max(eout, new_exponent)
        mout = mout // 2 ** (max_exponent - eout)
        current_result = mx_for_accumulation[i] @ mw_for_accumulation[i].transpose(-2, -1)
        current_result = reshape_to_block(current_result, x_config["dim_1"], w_config["dim_1"], x_config["parallism_dim_1"], w_config["parallism_dim_1"])
        current_result = current_result // 2 ** (max_exponent - new_exponent)
        mout += current_result
        eout = max_exponent

    # the shape of qout will be [batch_size, depth_in_n, depth_out_features, paral_n, paral_out_features]
    # the shape of mb will be [1, 1, out_features]
    # reshape mb to [1, 1, depth_out_features, 1, paral_out_features]
    # broad cast to [batch_size, depth_in_n, depth_out_features, paral_n, paral_out_features]

    # the shape of eout willbe [batch_size, depth_n, depth_out_features]
    # the shape of eb will be [1, 1, depth_out_featuers]
    
    # so i wish eb can map back to 
    out_config = config["out_config"]
    b_config = config["bias_config"]
    width_difference = x_config["width"] + w_config["width"] - 2 - (b_config["width"] -1)
    reshaped_mb = mb.reshape(1, 1, out_config["depth_dim_0"], 1, out_config["parallism_dim_0"])
    reshaped_eb = eb.reshape(1, 1, out_config["depth_dim_0"], 1, 1)
    mb_for_out = reshaped_mb // 2**(eout - reshaped_eb - width_difference)
    mout = mout + mb_for_out

    qout = reshape_back((mout / 2 **(x_config["width"]+w_config["width"] - 2 - eout)), x_config["dim_1"], w_config["dim_1"], x_config["parallism_dim_1"], w_config["parallism_dim_1"])
    qout = qout.reshape(batch_size, n, out_features)

    return qout

class MXIntLinear(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        q_config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert q_config is not None, "config is None!"
        self.in_features = in_features
        self.out_features = out_features
        self.q_config = q_config
        self.bypass = q_config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizer

    def forward(self, x: Tensor) -> Tensor:
        # an example of config
        unroll_in_features = self.q_config["data_in_parallelism"][1]
        unroll_out_features = self.q_config["data_out_parallelism"][1]
        unroll_n = self.q_config["data_in_parallelism"][0]
        in_features = self.in_features
        out_features = self.out_features
        n = x.shape[1]
        batch_size = x.shape[0]
        assert x.shape[2] == in_features, f"Input shape mismatch: {x.shape[2]} != {in_features}"

        self.config = {
            "x_config": {
                "width": self.q_config["data_in_width"],
                "exponent_width": self.q_config["data_in_exponent_width"],
                "parallism_dim_0": unroll_in_features,
                "parallism_dim_1": unroll_n,
                "depth_dim_0": in_features // unroll_in_features,
                "depth_dim_1": n // unroll_n,
                "dim_0": in_features,
                "dim_1": n,
            },
            "w_config": {
                "width": self.q_config["weight_width"],
                "exponent_width": self.q_config["weight_exponent_width"],
                "parallism_dim_0": unroll_in_features,
                "parallism_dim_1": unroll_out_features,
                "depth_dim_0": in_features // unroll_in_features,
                "depth_dim_1": out_features // unroll_out_features,
                "dim_0": in_features,
                "dim_1": out_features,
            },
            "bias_config": {
                "width": self.q_config["bias_width"],
                "exponent_width": self.q_config["bias_exponent_width"],
                "parallism_dim_0": unroll_out_features,
                "parallism_dim_1": 1,
                "depth_dim_0": out_features // unroll_out_features,
                "depth_dim_1": 1,
                "dim_0": out_features,
                "dim_1": 1,
            },
            "out_config": {
                "width": self.q_config["data_out_width"],
                "exponent_width": self.q_config["data_out_exponent_width"],
                "parallism_dim_0": unroll_out_features,
                "parallism_dim_1": unroll_n,
                "depth_dim_0": out_features // unroll_out_features,
                "depth_dim_1": n // unroll_n,
                "dim_0": out_features,
                "dim_1": n,
            },
            "round_bits": self.q_config.get("round_bits", 4),
        }
        # out = fast_linear(x, self.weight, self.bias, self.config)
        out = torch.nn.Linear(in_features, out_features, bias=True)(x)
        return out
