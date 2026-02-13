import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_tensor_bits_fp(tensor_shape: np.ndarray, width: int):
    return np.prod(tensor_shape) * width


def compute_tensor_bits_integer(tensor_shape: np.ndarray, width: int):
    return np.prod(tensor_shape) * width


def compute_tensor_bits_block_fp(
    tensor_shape: np.ndarray, width: int, exponent_width: int, block_size: np.ndarray
):
    if tensor_shape.size > block_size.size:
        block_size = np.append([1] * (tensor_shape.size - block_size.size), block_size)
    elif tensor_shape.size < block_size.size:
        block_size = block_size[-tensor_shape.ndim :]

    num_blocks = np.prod(np.ceil(tensor_shape / block_size))
    return num_blocks * np.prod(block_size) * width + num_blocks * exponent_width


def profile_linear_layer(
    quant_config: dict, in_features: int, out_features: int, bias: bool, batch_size: int
):
    """
    {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
    }
    """
    # logger.debug(
    #     f"quant_config = {quant_config},\nin_features = {in_features}, out_features = {out_features}, bias = {bias}, batch_size = {batch_size}"
    # )
    w_shape = np.array((in_features, out_features))
    b_shape = np.array((out_features,))
    x_shape = np.array((batch_size, in_features))

    # compute num of params, bias and activations
    num_params = in_features * out_features
    if bias:
        num_params += out_features
    num_xs = batch_size * in_features

    # compute param, bias and activation bits
    quant_arith = quant_config["name"]
    if quant_config.get("bypass", False):
        w_width = 32
        b_width = 32
        x_width = 32

        p_bits = compute_tensor_bits_fp(w_shape, w_width)
        if bias:
            p_bits += compute_tensor_bits_fp(b_shape, b_width)
        x_bits = compute_tensor_bits_fp(x_shape, x_width)
    else:
        w_width = quant_config["weight_width"]
        x_width = quant_config["data_in_width"]
        if bias:
            b_width = quant_config["bias_width"]
        match quant_arith:
            case "integer":
                p_bits = compute_tensor_bits_integer(w_shape, w_width)
                if bias:
                    p_bits += compute_tensor_bits_integer(b_shape, b_width)
                x_bits = compute_tensor_bits_integer(x_shape, x_width)
            case "block_fp":
                w_block_size = np.array(quant_config["weight_block_size"])
                if bias:
                    b_block_size = np.array(quant_config["bias_block_size"])
                x_block_size = np.array(quant_config["data_in_block_size"])

                p_bits = compute_tensor_bits_block_fp(
                    w_shape,
                    w_width,
                    quant_config["weight_exponent_width"],
                    w_block_size,
                )
                if bias:
                    p_bits += compute_tensor_bits_block_fp(
                        b_shape,
                        b_width,
                        quant_config["bias_exponent_width"],
                        b_block_size,
                    )
                x_bits = compute_tensor_bits_block_fp(
                    x_shape,
                    x_width,
                    quant_config["data_in_exponent_width"],
                    x_block_size,
                )
            case _:
                raise ValueError(f"Unknown quant_arith: {quant_arith}")
    # logger.debug(
    #     f"num_params = {num_params}, num_xs = {num_xs}, p_bits = {p_bits}, x_bits = {x_bits}"
    # )
    # x [batch_size, in_features], w [in_features, out_features], b [out_features]
    # flops = batch_size * out_features * (2 * in_features - 1) + in_features * out_features
    flops = batch_size * out_features * (2 * in_features - 1)
    if bias:
        flops += batch_size * out_features
    return {
        "num_params": np.rint(num_params).astype(np.int64),
        "num_acts": np.rint(num_xs).astype(np.int64),
        "param_bits": np.rint(p_bits).astype(np.int64),
        "act_bits": np.rint(x_bits).astype(np.int64),
        "flops": np.rint(flops).astype(np.int64),
    }


def profile_matmul_layer(quant_config: dict, data_in_0_size, data_in_1_size):
    """
    {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
    """

    x0_shape = np.array((data_in_0_size,))
    x1_shape = np.array((data_in_1_size,))
    num_xs = np.prod(x0_shape) + np.prod(x1_shape)

    quant_arith = quant_config["name"]
    num_params = 0

    param_bits = 0
    if quant_config.get("bypass", False):
        x0_width = x1_width = 32
        x_bits = compute_tensor_bits_fp(x0_shape, x0_width) + compute_tensor_bits_fp(
            x1_shape, x1_width
        )
    else:
        x0_width = quant_config["data_in_width"]
        x1_width = quant_config["data_in_width"]
        match quant_arith:
            case "integer":
                x_bits = compute_tensor_bits_integer(
                    x0_shape, x0_width
                ) + compute_tensor_bits_integer(x1_shape, x1_width)
            case "block_fp":
                x0_block_size = np.array(quant_config["data_in_block_size"])
                x1_block_size = np.array(quant_config["weight_block_size"])
                x_bits = compute_tensor_bits_block_fp(
                    x0_shape,
                    x0_width,
                    quant_config["data_in_exponent_width"],
                    x0_block_size,
                ) + compute_tensor_bits_block_fp(
                    x1_shape,
                    x1_width,
                    quant_config["weight_exponent_width"],
                    x1_block_size,
                )
            case _:
                raise ValueError(f"Unknown quant_arith: {quant_arith}")

    flops = data_in_0_size[0] * data_in_1_size[1] * (2 * data_in_0_size[1] - 1)
    return {
        "num_params": np.rint(num_params).astype(np.int64),
        "num_acts": np.rint(num_xs).astype(np.int64),
        "param_bits": np.rint(param_bits).astype(np.int64),
        "act_bits": np.rint(x_bits).astype(np.int64),
        "flops": np.rint(flops).astype(np.int64),
    }


def update_profile(profile, delta):
    profile["num_params"] += delta["num_params"]
    profile["num_acts"] += delta["num_acts"]
    profile["param_bits"] += delta["param_bits"]
    profile["act_bits"] += delta["act_bits"]
    profile["flops"] += delta["flops"]
    return profile


def _profile_opt_layer(
    layer_quant_config: dict,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    seq_len: int,
    bias: bool,
):
    """
    K = X W_k + b
    Q = X W_q + b
    V = X W_v + b

    A = Q K^T
    A = A V

    O = A W_o + b
    Y = O W_1 + b
    Y = Y W_2 + b

    """

    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
    }
    delta_list = []
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["q_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["k_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["v_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    for i in range(num_attention_heads):
        delta_list.append(
            profile_matmul_layer(
                layer_quant_config["self_attn"]["bmm_0"],
                data_in_0_size=(seq_len, hidden_size // num_attention_heads),
                data_in_1_size=(hidden_size // num_attention_heads, seq_len),
            )
        )
        delta_list.append(
            profile_matmul_layer(
                layer_quant_config["self_attn"]["bmm_1"],
                data_in_0_size=(seq_len, seq_len),
                data_in_1_size=(seq_len, hidden_size // num_attention_heads),
            )
        )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["out_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["fc1"],
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["fc2"],
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )

    for delta in delta_list:
        update_profile(profile, delta)
    return profile


def profile_opt_quantized(config, seq_len: int):
    """
    Profile opt quantized model

    Args:
        config (OPTQuantizedConfig): opt quantized config
        seq_len (int): sequence length
    """
    hidden_size = config.hidden_size
    intermediate_size = config.ffn_dim
    num_hidden_layers = config.num_hidden_layers

    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
    }

    for i in range(num_hidden_layers):
        layer_quant_config = config.quant_config[f"model_layer_{i}"]
        num_attention_heads = (
            config.num_attention_heads[i]
            if isinstance(config.num_attention_heads, (list, tuple))
            else config.num_attention_heads
        )
        update_profile(
            profile=profile,
            delta=_profile_opt_layer(
                layer_quant_config,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                seq_len=seq_len,
                bias=config.enable_bias,
            ),
        )
    return profile


def _profile_bert_layer(
    layer_quant_config: dict,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    seq_len: int,
    bias: bool,
):
    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
    }
    delta_list = []
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["attention"]["query"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["attention"]["key"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["attention"]["value"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )

    for i in range(num_attention_heads):
        delta_list.append(
            profile_matmul_layer(
                layer_quant_config["attention"]["matmul_0"],
                data_in_0_size=(seq_len, hidden_size // num_attention_heads),
                data_in_1_size=(hidden_size // num_attention_heads, seq_len),
            )
        )
        delta_list.append(
            profile_matmul_layer(
                layer_quant_config["attention"]["matmul_1"],
                data_in_0_size=(seq_len, seq_len),
                data_in_1_size=(seq_len, hidden_size // num_attention_heads),
            )
        )

    delta_list.append(
        profile_linear_layer(
            layer_quant_config["output"]["dense"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["intermediate"]["dense"],
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["output"]["dense"],
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )

    for delta in delta_list:
        update_profile(profile, delta)

    return profile


def profile_bert_quantized(config, seq_len: int):
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_hidden_layers = config.num_hidden_layers

    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
    }

    for i in range(num_hidden_layers):
        layer_quant_config = config.quant_config[f"model_layer_{i}"]
        num_attention_heads = (
            config.num_attention_heads[i]
            if isinstance(config.num_attention_heads, (list, tuple))
            else config.num_attention_heads
        )
        update_profile(
            profile=profile,
            delta=_profile_bert_layer(
                layer_quant_config,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                seq_len=seq_len,
                bias=True,
            ),
        )
    return profile


def _profile_llama_layer(
    layer_quant_config: dict,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    seq_len: int,
    bias: bool = False,
):
    """
    An example of quant_config for llama

    {
        "model_layer": {
            "self_attn": {
                "q_proj": {},
                "k_proj": {},
                "v_proj": {},
                "o_proj": {},
                "rotary_positional_encoding": {},
                "matmul_0": {},
                "matmul_1": {},
            },
            "mlp": {
                "gate_proj": {},
                "down_proj": {},
                "up_proj": {},
            },
        }
        "linear_default": {},
        "matmul_default": {},
    }
    """

    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
    }

    delta_list = []
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["q_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["k_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["v_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    for i in range(num_attention_heads):
        delta_list.append(
            profile_matmul_layer(
                layer_quant_config["self_attn"]["matmul_0"],
                data_in_0_size=(seq_len, hidden_size // num_attention_heads),
                data_in_1_size=(hidden_size // num_attention_heads, seq_len),
            )
        )
        delta_list.append(
            profile_matmul_layer(
                layer_quant_config["self_attn"]["matmul_1"],
                data_in_0_size=(seq_len, seq_len),
                data_in_1_size=(seq_len, hidden_size // num_attention_heads),
            )
        )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["o_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["mlp"]["gate_proj"],
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["mlp"]["down_proj"],
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["mlp"]["up_proj"],
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=bias,
            batch_size=seq_len,
        )
    )

    for delta in delta_list:
        update_profile(profile, delta)

    return profile


def profile_llama_quantized(config, seq_len: int):
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_hidden_layers = config.num_hidden_layers

    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
    }

    for i in range(num_hidden_layers):
        layer_quant_config = config.quant_config[f"model_layer_{i}"]
        num_attention_heads = (
            config.num_attention_heads[i]
            if isinstance(config.num_attention_heads, (list, tuple))
            else config.num_attention_heads
        )
        update_profile(
            profile=profile,
            delta=_profile_llama_layer(
                layer_quant_config,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                seq_len=seq_len,
                bias=False,
            ),
        )

    return profile


def get_model_profiler(profiler_name):
    match profiler_name:
        case "opt_quantized":
            return profile_opt_quantized
        case "bert_quantized":
            return profile_bert_quantized
        case "llama_quantized":
            return profile_llama_quantized
        case _:
            raise ValueError(f"Unknown profiler_name: {profiler_name}")
