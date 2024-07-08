#!/usr/bin/env python3

import os

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
import logging
from functools import partial
from einops import rearrange

import math

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import Timer

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner

# from mase_cocotb import Testbench, StreamDriver, StreamMonitor, mase_runner
from chop.nn.quantized import BertSelfAttentionInteger, fixed_softermax
#from chop.passes.graph.transforms.quantize.quantized_funcs import matmul_integer

from mase_cocotb.utils import fixed_preprocess_tensor

def exists(val):
    return val is not None
    
def default(val, d):
    return val if exists(val) else d

def get_positional_features_exponential(
    positions, features, seq_len, min_half_life=3.0
):
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(
        min_half_life, max_range, features, device=positions.device
    )
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions)


def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = (
        2 ** torch.arange(1, features + 1, device=positions.device).float()
    )
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(
        concentration
    ) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(
    positions, features, seq_len, stddev=None, start_mean=None, eps=1e-8
):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(
        start_mean, seq_len, features, device=positions.device
    )
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2
    probabilities = gamma_pdf(
        positions.float().abs()[..., None], concentration, rate
    )
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs

def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)
    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma,
    ]
    if feature_size % 6 == 0:
        feature_functions = feature_functions
    elif feature_size % 4 == 0:
        feature_functions = feature_functions[:2]
    else:
        feature_functions = feature_functions[:1]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size {feature_size} is not divisible"
            f"by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))
    print(embeddings)
    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    print(embeddings)
    return embeddings

def relative_shift_swin(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    h, _, window, t1, t2 = x.shape
    x = x.reshape(h, -1, window, t2, t1)
    x = x[:, :, :, 1:, :]
    x = x.reshape(h, -1, window, t1, t2 - 1)
    # up to this point: (i,j) in x represets the emb of dot product (i,i-j)
    return x[..., : ((t2 + 1) // 2)]

class SwinAttentionHelp(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        heads=8,
        dim_key=32,
        dropout=0.0,
        num_rel_pos_features=None,
        debug=False,
        padding_mode=True,
        window_specific_bias=False,
        seq_len=336,
    ):
        super().__init__()
        self.dim_key = dim_key
        self.heads = heads
        self.window_size = window_size

        self.to_qkv = nn.Linear(dim, dim_key * 3 * heads, bias=False)
        self.to_out = nn.Linear(dim_key * heads, dim)

        self.num_rel_pos_features = default(
            num_rel_pos_features, dim_key * heads
        )
        self.rel_pos_embedding = nn.Linear(
            self.num_rel_pos_features, dim_key * heads, bias=False
        )
        self.rel_content_bias = nn.Parameter(
            torch.randn(1, heads, 1, 1, dim_key)
        )
        rel_pos_bias_shape = (
            seq_len * 2 // window_size - 1 if window_specific_bias else 1
        )
        self.rel_pos_bias = nn.Parameter(
            torch.randn(1, heads, rel_pos_bias_shape, 1, dim_key)
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.pos_dropout = nn.Dropout(dropout)

        self.debug = debug
        self.padding_mode = padding_mode
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Sequential):
            for submodule in m.children():
                self._init_weights(submodule)
        elif isinstance(m, nn.ModuleList):
            for submodule in m:
                self._init_weights(submodule)

    def forward(self, x):
        b, n, c = x.shape
        original_n = n
        device = x.device
        remainder = n % self.window_size
        needs_padding = remainder > 0
        assert (
            n >= self.window_size
        ), f"the sequence {n} is too short for the window {self.window_size}"
        if self.padding_mode is False:
            assert needs_padding, (
                f"Sequence length ({n}) should be"
                f"divisibleby the window size ({self.window_size})."
            )
        else:
            if needs_padding:
                padding_size = self.window_size - remainder
                x = F.pad(x, (0, 0, 0, padding_size, 0, 0), value=0)
                mask = create_padding_mask(b, remainder, padding_size, device)
                n += padding_size

        #print("X: ", x)
        qkv = self.to_qkv(x)
        #print("weights:", self.to_qkv.weight.data)
        #print("qkv: ", qkv)
        
        qkv = rearrange(qkv, "b n (h d k) -> k b h n d", h=self.heads, k=3)
        #print("qkv rearranged: ", qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]

        print("Q: ", q)
        print("K: ", k)
        print("V: ", v)

        # Create sliding window indices
        window_indices = torch.arange(
            0, n - self.window_size + 1, self.window_size, device=device
        )
        q_windows = q[
            ...,
            window_indices.unsqueeze(-1)
            + torch.arange(self.window_size, device=device).unsqueeze(0),
            :,
        ]

        k_windows = k[
            ...,
            window_indices.unsqueeze(-1)
            + torch.arange(self.window_size, device=device).unsqueeze(0),
            :,
        ]

        v_windows = v[
            ...,
            window_indices.unsqueeze(-1)
            + torch.arange(self.window_size, device=device).unsqueeze(0),
            :,
        ]


        # position
        positions = get_positional_embed(
            self.window_size, self.num_rel_pos_features, device
        )
        positions = self.pos_dropout(positions)
        rel_k = self.rel_pos_embedding(positions)
        rel_k = rearrange(
            rel_k, "n (h d) -> h n d", h=self.heads, d=self.dim_key
        )
        # original rel_k is (h,windowSize, dimKey)
        # duplicate the rel_K for each window it should have shape
        # (h,numWindows,windowSize, dimKey)
        rel_k = rel_k.unsqueeze(1).repeat(1, q_windows.shape[2], 1, 1)

        print("sum: ", q_windows + self.rel_content_bias)

        k_windows = k_windows.transpose(-2, -1)
        content_attn = torch.matmul(
            q_windows + self.rel_content_bias,
            k_windows
            # q_windows,
            # k_windows,
        ) * (self.dim_key**-0.5)

        # calculate position attention
        rel_k = rel_k.transpose(-2, -1)

        rel_logits = torch.matmul(
            q_windows + self.rel_pos_bias,
            rel_k
            # q_windows,
            # rel_k,
        )
        # reshape position_attn to (b, h, n, w, w)
        position_attn = relative_shift_swin(rel_logits)

        attn = content_attn + position_attn
        if needs_padding:
            mask_value = -torch.finfo(attn.dtype).max
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn[:, :, -1, :, :] = attn[:, :, -1, :, :].masked_fill(
                mask, mask_value
            )
        print("attention", attn)
        attn = F.softmax(attn, dim=-1)
        print("softmax", attn)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v_windows)
        out = rearrange(out, "b h w n d -> b w n (h d)")
        out = self.to_out(out)
        out = self.proj_dropout(out)

        out = rearrange(out, "b w n d -> b (w n) d")
        return out[:, :original_n]


class FixedSwinAttentionTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        torch.manual_seed(0)

        self.model = SwinAttentionHelp(
            dim = 4,
            window_size = 4,
            heads = 1,
            dim_key = 4,
            dropout = 0,
            num_rel_pos_features = None,
            debug = False,
            padding_mode = True,
            window_specific_bias = False,
            seq_len = 4,
        )

        self.model.to_qkv.weight.data = (torch.randn(self.model.to_qkv.weight.shape))
        print(self.model.to_qkv.weight.data)
        #self.qkv_split_weights = torch.split(self.model.to_qkv.weight.data, 4, dim=0)

        #rearrange weights so that outputs match
        self.qkv_split_weights = []
        self.qkv_split_weights.append(self.model.to_qkv.weight.data[[0, 3, 6, 9], : ]) 
        self.qkv_split_weights.append(self.model.to_qkv.weight.data[[1, 4, 7, 10], :]) 
        self.qkv_split_weights.append(self.model.to_qkv.weight.data[[2, 5, 8, 11], :]) 

        self.rel_content_bias = self.model.rel_content_bias
        self.rel_content_bias = self.rel_content_bias.squeeze()
        self.rel_content_bias = self.rel_content_bias.unsqueeze(0)
        self.rel_content_bias = self.rel_content_bias.repeat(4,1)
        print("rel content bias", self.rel_content_bias)


        if self.get_parameter("HAS_BIAS") == 1:
            self.model.to_qkv.bias = nn.Parameter(torch.randn(self.model.to_qkv.out_features))
            self.qkv_split_bias = torch.split(self.model.to_qkv.bias.data, 4, dim=0)
        else:
            self.model.to_qkv.bias = nn.Parameter(torch.zeros(self.model.to_qkv.out_features))


        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )

        # * Weight drivers
        self.weight_query_driver = StreamDriver(
            dut.clk, dut.weight_query, dut.weight_query_valid, dut.weight_query_ready
        )
        self.weight_key_driver = StreamDriver(
            dut.clk, dut.weight_key, dut.weight_key_valid, dut.weight_key_ready
        )
        self.weight_value_driver = StreamDriver(
            dut.clk, dut.weight_value, dut.weight_value_valid, dut.weight_value_ready
        )

        self.bias_con_driver = StreamDriver(
            dut.clk, dut.bias_content, dut.bias_content_valid, dut.bias_content_ready
        )

        self.bias_pos_driver = StreamDriver(
            dut.clk, dut.bias_position, dut.bias_position_valid, dut.bias_position_ready
        )

        if self.get_parameter("HAS_BIAS") == 1:
            self.bias_query_driver = StreamDriver(
                dut.clk, dut.biasquery_, dut.bias_query_valid, dut.bias_query_ready
            )
            self.bias_key_driver = StreamDriver(
                dut.clk, dut.bias_key, dut.bias_key_valid, dut.bias_key_ready
            )
            self.bias_value_driver = StreamDriver(
                dut.clk, dut.bias_value, dut.bias_value_valid, dut.bias_value_ready
            )
            self.bias_query_driver.log.setLevel(logging.DEBUG)
            self.bias_key_driver.log.setLevel(logging.DEBUG)
            self.bias_value_driver.log.setLevel(logging.DEBUG)

        self.data_out_0_monitor = StreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=False,
        )

        

        # Set verbosity of driver and monitor loggers to debug
        self.data_in_0_driver.log.setLevel(logging.DEBUG)
        self.weight_query_driver.log.setLevel(logging.DEBUG)
        self.weight_key_driver.log.setLevel(logging.DEBUG)
        self.weight_value_driver.log.setLevel(logging.DEBUG)
        self.bias_con_driver.log.setLevel(logging.DEBUG)
        self.bias_pos_driver.log.setLevel(logging.DEBUG)
        self.data_out_0_monitor.log.setLevel(logging.DEBUG)

    def generate_inputs(self, batch_size=1):
        return torch.randn(
            (
                batch_size,
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_1"),
                self.get_parameter("DATA_IN_0_TENSOR_SIZE_DIM_0"),
            )
        )

    def generate_weights(self, batch_size=1):
        return torch.randn(
            (
                batch_size,
                self.get_parameter("WEIGHT_TENSOR_SIZE_DIM_0"),
                self.get_parameter("WEIGHT_TENSOR_SIZE_DIM_1"),
            )
        )

    def generate_bias(self, batch_size=1):
        return torch.randn(
            (
                batch_size,
                self.get_parameter("BIAS_TENSOR_SIZE_DIM_0"),
                self.get_parameter("BIAS_TENSOR_SIZE_DIM_1"),
            )
        )

    async def run_test(self):
        await self.reset()
        self.log.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1

        inputs = self.generate_inputs()
        exp_out = self.model(inputs)

        #print("bias: ", self.model.to_qkv.bias)

        weights = self.model.to_qkv.weight.data
        #print(weights)
        #print("k weights: ", self.qkv_split_weights[1])
        #print("inputs: ", inputs)

        #print("k transpose: ", torch.matmul(inputs, self.qkv_split_weights[1].t()))
        #print("k: ", torch.matmul(inputs, self.qkv_split_weights[1]))
        


        #print("Inputs: ", inputs)
        inputs = fixed_preprocess_tensor(
            tensor=inputs,
            q_config={
                "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                "frac_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
            ],
        )
        self.data_in_0_driver.load_driver(inputs)

        for index, projection in enumerate(["query", "key", "value"]):
            if self.get_parameter("WEIGHTS_PRE_TRANSPOSED") == 1:
                weights = self.qkv_split_weights[index].transpose(0, 1)
            else:
                weights = self.qkv_split_weights[index]

            

            self.log.info(f"Processing {projection} weights: {weights}")
            weights = fixed_preprocess_tensor(
                tensor=weights,
                q_config={
                    "width": self.get_parameter("WEIGHT_PRECISION_0"),
                    "frac_width": self.get_parameter("WEIGHT_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter("WEIGHT_PARALLELISM_DIM_1"),
                    self.get_parameter("WEIGHT_PARALLELISM_DIM_0"),
                ],
            )
            getattr(self, f"weight_{projection}_driver").load_driver(weights)

            # * Load the bias driver
            if self.get_parameter("HAS_BIAS") == 1:
                bias = self.qkv_split_bias[index]
                self.log.info(f"Processing {projection} bias: {bias}")
                bias = fixed_preprocess_tensor(
                    tensor=bias,
                    q_config={
                        "width": self.get_parameter("BIAS_PRECISION_0"),
                        "frac_width": self.get_parameter("BIAS_PRECISION_1"),
                    },
                    parallelism=[
                        self.get_parameter("BIAS_PARALLELISM_DIM_1"),
                        self.get_parameter("BIAS_PARALLELISM_DIM_0"),
                    ],
                )
                getattr(self, f"bias_{projection}_driver").load_driver(bias)

            content_bias = fixed_preprocess_tensor(
                    tensor=self.rel_content_bias,
                    q_config={
                        "width": self.get_parameter("BIAS_PRECISION_0"),
                        "frac_width": self.get_parameter("BIAS_PRECISION_1"),
                    },
                    parallelism=[
                        self.get_parameter("BIAS_PARALLELISM_DIM_1"),
                        self.get_parameter("BIAS_PARALLELISM_DIM_0"),
                    ],
                )
            self.bias_con_driver.load_driver(content_bias)
            
        
        self.log.info(f"Processing outputs: {exp_out}")
        outs = fixed_preprocess_tensor(
            tensor=exp_out,
            q_config={
                "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                "frac_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1"),
                self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"),
            ],
        )
        self.data_out_0_monitor.load_monitor(outs)

        await Timer(1, units="ms")

        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    tb = FixedSwinAttentionTB(dut)
    await tb.run_test()


def get_config(kwargs={}):
    config = {
        "NUM_HEADS": 1,
        "ACTIVATION": 0,
        "DATA_IN_0_TENSOR_SIZE_DIM_0": 4,
        "DATA_IN_0_TENSOR_SIZE_DIM_1": 4,
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 2,
        "DATA_IN_0_PRECISION_0": 16,
        "DATA_IN_0_PRECISION_1": 8,
        "WEIGHTS_PRE_TRANSPOSED": 1,
        "WEIGHT_TENSOR_SIZE_DIM_0": 4,
        "WEIGHT_TENSOR_SIZE_DIM_1": 4,
        "WEIGHT_PARALLELISM_DIM_0": 2,
        "WEIGHT_PARALLELISM_DIM_1": 2,
        "WEIGHT_PRECISION_0": 16,
        "WEIGHT_PRECISION_1": 8,
        "HAS_BIAS": 0,
        "BIAS_TENSOR_SIZE_DIM_0": 4,
        "BIAS_TENSOR_SIZE_DIM_1": 4,
        "BIAS_PARALLELISM_DIM_0": 2,
        "BIAS_PARALLELISM_DIM_1": 2,
        "BIAS_PRECISION_0": 16,
        "BIAS_PRECISION_1": 8,
        "DATA_OUT_0_TENSOR_SIZE_DIM_0": 4,
        "DATA_OUT_0_TENSOR_SIZE_DIM_1": 4,
        "DATA_OUT_0_PARALLELISM_DIM_0": 2,
        "DATA_OUT_0_PARALLELISM_DIM_1": 2,
        "DATA_OUT_0_PRECISION_0": 16,
        "DATA_OUT_0_PRECISION_1": 8,
    }
    config.update(kwargs)
    return config


def test_fixed_linear_smoke():
    """
    Some quick tests to check if the module is working.
    """
    mase_runner(trace=True, module_param_list=[get_config()], skip_build=False)


if __name__ == "__main__":
    test_fixed_linear_smoke()
