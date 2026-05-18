"""Quantization transform pass.

Step-1 integration note (Llama phase split):
- We keep this pass as a thin wiring layer.
- Runtime phase behavior remains inside quantized modules.
- This pass only normalizes/forwards configs to preserve backward compatibility.
"""

from copy import deepcopy

import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from chop.nn.quantized.modules import quantized_module_map
from chop.nn.quantized.modules.phase_config import normalize_phase_q_config
from chop.nn.quantized.modules.phase_context import (
    infer_phase_from_decoder_layer_inputs,
    set_active_phase,
    set_decode_policy,
)
from ...module_modify_helper import replace_by_name, instantiate_module
from ...state_dict_map import match_a_pattern, check_is_huggingface_model


def _prepare_module_config(config: dict, postfix: str) -> dict:
    """Prepare module config before module instantiation.

    Why this helper exists:
    - `quantize` pass historically expected a flat config dict.
    - Step-1 phase split introduces optional `{prefill, decode}` structure.
    - We normalize only for phase-aware quantizers so other quantizers remain
      behavior-identical.
    """

    cfg = deepcopy(config)
    phase_aware_postfixes = {"mxfp", "mxint", "minifloat"}
    if postfix in phase_aware_postfixes:
        return normalize_phase_q_config(cfg)
    return cfg


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def _has_quantized_llama_modules(network) -> bool:
    """Return True when network contains quantized Llama modules.

    Why name-based detection:
    - It avoids additional direct imports from quantized module files.
    - It keeps this pass decoupled from specific class symbols while still
      matching the module-replacement products used in step-1.
    """

    quantized_llama_class_names = {
        "LlamaAttentionMXFP",
        "LlamaAttentionMXInt",
        "LlamaMLPMXFP",
        "LlamaMLPMXInt",
        "LlamaRMSNormMinifloat",
    }
    for module in network.modules():
        if module.__class__.__name__ in quantized_llama_class_names:
            return True
    return False


def _llama_decoder_layer_phase_pre_hook(module, args, kwargs):
    """Set runtime phase before decoder-layer body executes.

    Hook timing is critical: it runs before `input_layernorm`, ensuring modules
    that execute before attention still observe correct phase in step-1.
    """

    phase = infer_phase_from_decoder_layer_inputs(args, kwargs)
    set_active_phase(phase)
    # Step-1 invariant: decode path is always full-precision.
    set_decode_policy("fp_only")
    return None


def _install_llama_phase_pre_hooks(network) -> None:
    """Install idempotent phase pre-hooks on all Llama decoder layers."""

    if not _has_quantized_llama_modules(network):
        return

    for module in network.modules():
        if not isinstance(module, LlamaDecoderLayer):
            continue
        if getattr(module, "_mase_phase_hook_installed", False):
            continue
        module.register_forward_pre_hook(_llama_decoder_layer_phase_pre_hook, with_kwargs=True)
        module._mase_phase_hook_installed = True


def quantize_by_type(network, pass_args):
    for type_name, config in pass_args.items():
        n_m = {}
        for n, m in network.named_modules():
            n_m[n] = m

        if type_name == "linear":
            module = torch.nn.Linear
        elif type_name == "conv2d":
            module = torch.nn.Conv2d
        else:
            raise ValueError(f"{type_name} is not supported!")
        config = deepcopy(config["config"])
        postfix = config.pop("name")
        config = _prepare_module_config(config, postfix)
        for n, m in n_m.items():
            if isinstance(m, module):
                new_m = instantiate_module(
                    m, postfix, quantized_module_map, {"config": config}
                )
                network = replace_by_name(network, n, new_m)
    return network


def quantize_by_name(network, pass_args):
    is_huggingface_model = check_is_huggingface_model(network)

    quantize_names = pass_args.keys()
    n_m = {}
    for n, m in network.named_modules():
        n_m[n] = m
    for n, m in n_m.items():
        if n in quantize_names:
            quan_config = deepcopy(pass_args[n]["config"])
            postfix = quan_config.pop("name")
            quan_config = _prepare_module_config(quan_config, postfix)

            additional_module_args = (
                {"config": quan_config, "network_config": network.config}
                if is_huggingface_model
                else {"config": quan_config}
            )

            new_m = instantiate_module(
                m, postfix, quantized_module_map, additional_module_args
            )
            network = replace_by_name(network, n, new_m)
    return network


def quantize_by_regex_name(network, pass_args):
    is_huggingface_model = check_is_huggingface_model(network)

    patterns = list(pass_args.keys())
    n_m = {}
    for n, m in network.named_modules():
        n_m[n] = m

    for n, m in n_m.items():
        matched_pattern = match_a_pattern(n, patterns)
        if not matched_pattern:
            continue

        quan_config = deepcopy(pass_args[matched_pattern]["config"])
        postfix = quan_config["name"]
        quan_config = _prepare_module_config(quan_config, postfix)

        additional_module_args = (
            {"config": quan_config, "network_config": network.config}
            if is_huggingface_model
            else {"config": quan_config}
        )

        new_m = instantiate_module(
            m, postfix, quantized_module_map, additional_module_args
        )
        network = replace_by_name(network, n, new_m)

    return network


def quantize_module_transform_pass(network, pass_args):
    """
    Apply quantization transformation to the given nn.Module.

    :param network: The input network to be transformed.
    :type network: torch.nn.Module

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    Examples pass_args:

    .. code-block:: python

        pass_args = {
            "by": "type", # quantize by type, name, or regex_name
            "default": {"config": {"name": None}}, # default config, this would be used for any node that does not have a specific config
            "linear": {
                "config": {
                    "name": "integer",  # quantization scheme name supported are ["integer", "fixed" (equivalent to integer), "lutnet" (dev mode), "logicnets" (dev mode), "binary", "binary_residual", "ternary", "minifloat_ieee", "minifloat_denorm", "log", "block_fp", "block_minifloat", "block_log"]
                    # data
                    "data_in_width": 8,
                    "data_in_frac_width": 4,
                    # weight
                    "weight_width": 8,
                    "weight_frac_width": 4,
                    # bias
                    "bias_width": 8,
                    "bias_frac_width": 4,
                }
            },
        }

    :return: The transformed torch.nn.Module.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.

    """
    # Defensive copy avoids mutating caller-owned pass_args, which is
    # important for reproducible experiment runners that reuse config dicts.
    pass_args = deepcopy(pass_args)

    # GPTQ pre-pass: quantize linear weights before module replacement.
    gptq_config = pass_args.pop("gptq", None)
    if gptq_config is not None:
        from ..gptq import run_gptq

        network = run_gptq(network, gptq_config)

    by = pass_args.pop("by")
    match by:
        case "type":
            network = quantize_by_type(network, pass_args)
        case "name":
            network = quantize_by_name(network, pass_args)
        case "regex_name":
            network = quantize_by_regex_name(network, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # Install phase hooks only after module replacement, so detection sees
    # quantized Llama modules rather than original HF modules.
    _install_llama_phase_pre_hooks(network)

    return network, {}
