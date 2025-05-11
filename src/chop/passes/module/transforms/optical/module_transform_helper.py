import torch
import torch.nn as nn
import math
from functools import reduce, partial
from copy import deepcopy
import logging
import inspect

from chop.passes.module.module_modify_helper import (
    get_module_by_name,
    set_module_by_name,
)
from chop.passes.module.state_dict_map import SPECIAL_CONVERT_PATTERNS

from transformers.models.roberta.modeling_roberta import (
    RobertaSelfAttention,
    RobertaSdpaSelfAttention,
    RobertaClassificationHead,
    RobertaIntermediate,
    RobertaOutput,
    RobertaSelfOutput,
)

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
)

from transformers.models.bert.modeling_bert import (
    BertSdpaSelfAttention,
    BertSelfAttention,
)

from transformers.models.bert.configuration_bert import BertConfig



bert_prefix_map = {
    BertSdpaSelfAttention: "bert_self_attention",
    BertSelfAttention: "bert_self_attention",
}

def check_module_instance(module, prefix_map):
    """
    Check if the given module is an instance of any class in the prefix_map. If it is, return the corresponding prefix.
    Args:
        module (object): The module to check.
        prefix_map (dict): A dictionary where keys are classes and values are prefixes.
    Returns:
        tuple: A tuple containing a boolean indicating if the module is an instance of any class in the prefix_map,
               and the corresponding prefix if it is an instance, otherwise None.
    """
    for cls, name in prefix_map.items():
        if isinstance(module, cls):
            return True, name
    return False, None

def replace_by_name_optical(network, module_name: str, new_module, target_name):

    original = get_module_by_name(network, module_name)
    if target_name == "linear_morr_full":
        updated_module = weight_replacement_full_linear_optical(original, new_module)
    elif target_name in ["linear_morr", "linear_morr_triton", "linear_morr_triton_mem"]:
        updated_module = weight_replacement_circulant_linear_optical(original, new_module)
    elif target_name in ["bert_self_attention_morr"]:
        updated_module = weight_replacement_circulant_bert_attention(original, new_module)
    else:
        raise NotImplementedError(f"weight replacement function for the optical module {target_name} not implemented")
    
    network = set_module_by_name(network, module_name, updated_module)

    return network


def weight_replacement_full_linear_optical(original, new_module):
    if isinstance(original, nn.Linear):
        return weight_replacement_linear_optical(original, new_module)
    elif isinstance(original, nn.Conv2d):
        return weight_replacement_conv2d_optical(original, new_module)
    else:
        raise NotImplementedError(
            "weight replacement function for the optical module not implemented"
        )

def weight_replacement_linear_optical(linear_layer, morr_layer):
    """
    Replace the weights of AllPassMORRLinear (morr_layer) with those from a standard nn.Linear (linear_layer).
    Focuses only on weight copying (no bias copying).
    """
    # Extract dimensions
    out_features = morr_layer.out_features
    in_features = morr_layer.in_features
    miniblock = morr_layer.miniblock
    grid_dim_x = morr_layer.grid_dim_x
    grid_dim_y = morr_layer.grid_dim_y
    in_features_pad = morr_layer.in_features_pad
    
    # Get the weights from the standard linear layer
    standard_weights = linear_layer.weight.data  # [out_features, in_features]
    
    # Ensure the shapes match
    assert standard_weights.shape[0] == out_features, "Output feature dimensions don't match"
    assert standard_weights.shape[1] == in_features, "Input feature dimensions don't match"
    
    # Pad the standard weights to match in_features_pad
    if in_features_pad > in_features:
        padded_weights = torch.zeros(out_features, in_features_pad, 
                                    device=standard_weights.device, 
                                    dtype=standard_weights.dtype)
        padded_weights[:, :in_features] = standard_weights
        standard_weights = padded_weights # [out_features, in_features_pad]
    
    # Reshape to match the MORR structure [grid_dim_y, grid_dim_x, miniblock]
    assert grid_dim_y == out_features, "grid_dim_y does not match out_features"
    assert grid_dim_x * miniblock == in_features_pad, "grid_dim_x * miniblock does not match in_features_pad"
    
    reshaped_weights = standard_weights.reshape(grid_dim_y, grid_dim_x, miniblock)
    
    # Copy the weights to the MORR layer
    with torch.no_grad():
        morr_layer.weight.data.copy_(reshaped_weights)
    
    return morr_layer

def weight_replacement_circulant_linear_optical(x, y):
    """
    Replace the weights of AllPassMORRCirculantLinear (y) with those from a standard nn.Linear (x).
    Focuses only on weight copying (no bias copying).
    """

    # Fetch original linear weight [out_features, in_features]
    W = x.weight.data  # [out_features, in_features]

    # Grab dimensions and zero-pad if needed
    out_features_pad = y.out_features_pad  # padded out_features in y
    in_features_pad = y.in_features_pad  # padded in_features in y
    miniblock = y.miniblock
    grid_dim_y = y.grid_dim_y
    grid_dim_x = y.grid_dim_x

    # Construct padded weight tensor
    W_padded = W.new_zeros((out_features_pad, in_features_pad))
    W_padded[: W.size(0), : W.size(1)] = W

    # Takes the mean across the miniblock slice.
    new_weight = W.new_zeros((grid_dim_y, grid_dim_x, miniblock)) # [grid_dim_y, grid_dim_x, miniblock]

    # Fill new_weight by averaging the corresponding sub-blocks in W_padded
    # original miniblock: [k, k] new miniblock: [k, 1]
    with torch.no_grad():
        for p in range(grid_dim_y):
            for q in range(grid_dim_x):
                for k in range(miniblock):
                    row_idx = p * miniblock + k # The row in W_padded:
                    col_start = q * miniblock # The columns in W_padded:
                    col_end = (q + 1) * miniblock
                    block = W_padded[row_idx, col_start:col_end]

                    new_weight[p, q, k] = block.mean()

        bound = 1 / math.sqrt(miniblock)
        new_weight = torch.rand((grid_dim_y, grid_dim_x, miniblock), 
                                device=W.device, 
                                dtype=W.dtype) * 2 * bound - bound
        # Copy the result into y.weight
        y.load_parameters({"weight": new_weight})

    return y


def weight_replacement_conv2d_optical(x, y):
    """
    Replace the weights (and bias, if present) of a standard nn.Conv2d (x)
    into an AllPassMORRCirculantConv2d (y).

    Args:
        x (nn.Conv2d):    A standard PyTorch Conv2d module
        y (AllPassMORRCirculantConv2d): An already-constructed optical Conv2d
                                        module into which we copy weights/bias.
    """
    with torch.no_grad():
        # 1) Copy bias (if both x and y actually have one).
        if x.bias is not None and y.bias is not None:
            y.bias.copy_(x.bias)

        # 2) Flatten nn.Conv2d's weight => shape [out_channels, in_channels*kernel_h*kernel_w]
        w_flat = x.weight.data.view(x.out_channels, -1)

        # 3) Zero-pad to match (out_channels_pad, in_channels_pad)
        outC_pad = y.out_channels_pad  # == y.grid_dim_y * y.miniblock
        inC_pad = y.in_channels_pad  # == y.grid_dim_x * y.miniblock

        W = torch.zeros(outC_pad, inC_pad, device=w_flat.device, dtype=w_flat.dtype)
        # Copy as many channels/elements as we have
        W[: x.out_channels, : w_flat.size(1)] = w_flat

        # 4) Reshape into blocks => shape [p, miniblock, q, miniblock]
        p = y.grid_dim_y
        q = y.grid_dim_x
        k = y.miniblock
        W_blocks = W.view(p, k, q, k)  # => [p, k, q, k]

        # 5) For each p,q block, extract the "first column" of size 'k' and place it in y.weight
        #    That is, for a k x k sub-block, we interpret sub_block[:,0] as the "circulant first column".
        for i in range(p):
            for j in range(q):
                sub_block = W_blocks[i, :, j, :]  # shape [k, k]
                y.weight.data[i, j, :] = sub_block[:, 0]

    # Done. At this point, y.weight and y.bias (if present) have been overwritten
    # with a simple block-circulant approximation of x's parameters.
    return y

def weight_replacement_circulant_bert_attention(original, new_module):
    for name in ("query", "key", "value"):
        src_linear = getattr(original, name)
        dst_linear = getattr(new_module, name)
        with torch.no_grad():
            dst_linear.weight.copy_(src_linear.weight)
            if src_linear.bias is not None:
                dst_linear.bias.copy_(src_linear.bias)
    
    return new_module


def instantiate_optical_module(module, postfix, module_map, additional_module_args):
    is_bert, bert_layer_name = check_module_instance(module, bert_prefix_map)

    module_args = additional_module_args["config"]
    additional_args = additional_module_args["additional"]
    network_args = additional_module_args.get("network_config", None)

    if isinstance(module, torch.nn.Linear):
        module = instantiate_optical_linear(module, postfix, module_map, module_args, additional_args)
    elif isinstance(module, torch.nn.Conv2d):
        module = instantiate_optical_conv2d(module, postfix, module_map, module_args)
    elif is_bert:
        module = instantiate_optical_bert_module(
            module, postfix, bert_layer_name, module_map, module_args,
        )
    else:
        raise ValueError(f"{module} is not supported.")
    return module

def instantiate_optical_linear(module, postfix, module_map, additional_module_args, additional_args):
    linear_cls = module_map[f"linear_{postfix}"]
    has_bias = not (module.bias is None)

    # TODO: some transformed modules have "config" as an argument then extract the additional_module_args from it. Some directly take the additional_module_args.
    # Need to handle this better
    if "config" in inspect.signature(linear_cls.__init__).parameters:
        linear = linear_cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=has_bias,
            config=additional_module_args,
        )
    else:
        linear = linear_cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=has_bias,
            **additional_module_args,
        )
        
    # extra handling for morr optical module
    enable_thermal_crosstalk = additional_args.get("thermal_crosstalk", False)
    enable_phase_noise = additional_args.get("phase_noise", False)
    enable_trainable_morr_scale = additional_args.get("trainable_morr_scale", False)
    enable_trainable_morr_bias = additional_args.get("trainable_morr_bias", False)

    if enable_thermal_crosstalk:
        linear.enable_crosstalk()
        linear.set_crosstalk_coupling_matrix(
            additional_args.get("coupling_factor", 0.04),
            additional_args.get("drop_perc", 0.0),
        )
    
    if enable_phase_noise:
        linear.enable_phase_variation()
        phase_noise_std = additional_args.get("phase_noise_std", 0.04)
        linear.set_phase_variation(phase_noise_std)
    
    if enable_trainable_morr_scale:
        linear.enable_trainable_morr_scale()
    
    if enable_trainable_morr_bias:
        linear.enable_trainable_morr_bias()
    
    return linear

def instantiate_optical_conv2d(module, postfix, module_map, additional_module_args):
    conv2d_cls = module_map[f"conv2d_{postfix}"]
    has_bias = not (module.bias is None)
    # TODO: some transformed modules have "config" as an argument then extract the additional_module_args from it. Some directly take the additional_module_args.
    # Need to handle this better
    if "config" in inspect.signature(conv2d_cls.__init__).parameters:
        conv2d = conv2d_cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=has_bias,
            padding_mode=module.padding_mode,
            config=additional_module_args,
        )
    else:
        conv2d = conv2d_cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=has_bias,
            padding_mode=module.padding_mode,
            **additional_module_args,
        )
    return conv2d

def instantiate_optical_bert_module(
    module, postfix, prefix, module_map, module_args,
):
    bert_cls = module_map[f"{prefix}_{postfix}"]

    bert_module = bert_cls(
        config=BertConfig(
            hidden_size=module.query.in_features,
            num_attention_heads=module.num_attention_heads,
            attention_head_size=module.attention_head_size,
            attention_probs_dropout_prob=module.dropout_prob,
            is_decoder=False,
        ),
        morr_config=module_args,
    )
    return bert_module