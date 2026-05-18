"""Quantization transform pass.

Llama phase-split integration note:
- This pass remains a thin wiring layer.
- Runtime phase execution stays in quantized modules.
- This pass only normalizes configs and writes a consistent runtime decode
  policy so downstream modules do not diverge silently.
"""

from copy import deepcopy
from functools import partial

import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from chop.nn.quantized.modules import quantized_module_map
from chop.nn.quantized.modules.phase_config import normalize_phase_q_config
from chop.nn.quantized.modules.phase_context import (
    infer_runtime_phase_from_decoder_layer_inputs,
    set_runtime_phase,
    set_runtime_decode_policy,
)
from ...module_modify_helper import replace_by_name, instantiate_module
from ...state_dict_map import match_a_pattern, check_is_huggingface_model


_LLAMA_PHASE_CONTEXT_POLICY_MODULE_CLASS_NAMES = {
    "LlamaAttentionMXFP",
    "LlamaAttentionMXInt",
    "LlamaMLPMXFP",
    "LlamaMLPMXInt",
    "LlamaRMSNormMinifloat",
    "LinearMXFP",
    "LinearMXInt",
}
"""Class names that participate in Llama phase-policy runtime wiring.

Why include LinearMX*:
- Linear modules consume runtime phase/decode policy during forward.
- Hook installation must trigger even for linear-only quantization configs.
- Reusing one list for detection + policy inference prevents drift.
"""


def _normalize_quantize_module_config(config: dict, postfix: str) -> dict:
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


def _has_llama_quantized_runtime_modules(network) -> bool:
    """Return True when network contains quantized Llama modules.

    Why name-based detection:
    - It avoids additional direct imports from quantized module files.
    - It keeps this pass decoupled from specific class symbols while still
      matching the module-replacement products used in step-1.
    """

    for module in network.modules():
        if (
            module.__class__.__name__
            in _LLAMA_PHASE_CONTEXT_POLICY_MODULE_CLASS_NAMES
        ):
            return True
    return False


def _infer_llama_decode_policy_from_quantized_modules(network) -> str:
    """Infer a single decode policy from quantized Llama modules.

    Why this is fail-fast:
    - Phase context is global per forward call.
    - If different Llama quantized modules disagree on decode policy, runtime
      behavior becomes order-dependent and difficult to debug.
    """

    observed_policies = set()
    for module in network.modules():
        if (
            module.__class__.__name__
            not in _LLAMA_PHASE_CONTEXT_POLICY_MODULE_CLASS_NAMES
        ):
            continue
        policy = getattr(module, "decode_policy", None)
        if policy is None:
            continue
        if policy not in ("fp_only", "quantized"):
            raise ValueError(
                f"Unsupported decode_policy={policy!r} on "
                f"{module.__class__.__name__}."
            )
        observed_policies.add(policy)

    if not observed_policies:
        return "fp_only"
    if len(observed_policies) > 1:
        raise ValueError(
            "Mixed decode policies detected across quantized Llama modules: "
            f"{sorted(observed_policies)}. Use a single decode_policy."
        )
    return next(iter(observed_policies))


def _llama_phase_context_pre_hook(module, args, kwargs, decode_policy):
    """Set runtime phase before decoder-layer body executes.

    Hook timing is critical: it runs before `input_layernorm`, ensuring modules
    that execute before attention still observe correct phase in step-1.
    """

    phase = infer_runtime_phase_from_decoder_layer_inputs(args, kwargs)
    set_runtime_phase(phase)
    # Keep decode policy explicit in context so all downstream modules apply
    # the same policy in this forward call.
    set_runtime_decode_policy(decode_policy)
    return None


def _install_llama_phase_context_pre_hooks(network) -> None:
    """Install idempotent phase pre-hooks on all Llama decoder layers."""

    if not _has_llama_quantized_runtime_modules(network):
        return
    decode_policy = _infer_llama_decode_policy_from_quantized_modules(network)

    for module in network.modules():
        if not isinstance(module, LlamaDecoderLayer):
            continue
        if getattr(module, "_mase_phase_hook_installed", False):
            prev_policy = getattr(module, "_mase_phase_hook_decode_policy", None)
            if prev_policy is not None and prev_policy != decode_policy:
                raise ValueError(
                    "Llama phase hook already installed with decode_policy="
                    f"{prev_policy!r}, but current network resolves to "
                    f"{decode_policy!r}. Rebuild model to avoid mixed policies."
                )
            continue
        module.register_forward_pre_hook(
            partial(_llama_phase_context_pre_hook, decode_policy=decode_policy),
            with_kwargs=True,
        )
        module._mase_phase_hook_installed = True
        module._mase_phase_hook_decode_policy = decode_policy


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
        config = _normalize_quantize_module_config(config, postfix)
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
            quan_config = _normalize_quantize_module_config(quan_config, postfix)

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
        quan_config = _normalize_quantize_module_config(quan_config, postfix)

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
    _install_llama_phase_context_pre_hooks(network)

    return network, {}
