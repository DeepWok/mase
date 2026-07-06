"""Quantization transform pass.

Phase-split integration (decode-side disaggregated serving):
- This pass remains a thin wiring layer; runtime phase execution stays in
  the quantized modules.
- After module replacement it installs idempotent pre-hooks on every
  decoder layer (Llama / Qwen3) that write the runtime phase (``prefill`` /
  ``decode``) into the shared phase context before the layer body executes,
  so all downstream phase-aware modules observe a consistent phase.
"""

import logging
from copy import deepcopy
from functools import partial

import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from chop.nn.quantized.modules import quantized_module_map
from chop.nn.quantized.modules.phase_context import (
    infer_runtime_phase_from_decoder_layer_inputs,
    set_runtime_phase,
    set_runtime_decode_policy,
)
from ...module_modify_helper import replace_by_name, instantiate_module
from ...state_dict_map import match_a_pattern, check_is_huggingface_model

# Decoder-layer classes that get runtime phase pre-hooks. Extending phase
# support to a new architecture only requires phase-aware modules plus an
# entry here.
_PHASE_HOOKED_DECODER_LAYERS = (LlamaDecoderLayer, Qwen3DecoderLayer)


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def _iter_phase_aware_modules(network):
    """Yield modules that resolve their quantisation config per phase.

    Detection is capability-based: every phase-aware module writes its
    normalized config to ``phase_q_config``, so new phase-aware modules
    participate without registry edits.
    """

    for module in network.modules():
        if hasattr(module, "phase_q_config"):
            yield module


def _infer_runtime_decode_policy(network) -> str | None:
    """Summarize the network's decode policy for the runtime context.

    Returns ``None`` when the network contains no phase-aware modules (no
    hooks needed). Policies may legitimately differ per module (e.g. a
    decode-quantised linear next to an always-FP rms_norm); each module
    obeys its own config, so the context policy is a summary: ``quantized``
    if any module quantises decode, else ``fp_only``.
    """

    saw_phase_module = False
    for module in _iter_phase_aware_modules(network):
        saw_phase_module = True
        if getattr(module, "decode_policy", None) == "quantized":
            return "quantized"
    return "fp_only" if saw_phase_module else None


def _phase_context_pre_hook(module, args, kwargs, decode_policy):
    """Set runtime phase before the decoder-layer body executes.

    Hook timing is critical: it runs before ``input_layernorm``, ensuring
    modules that execute before attention still observe the correct phase.
    """

    phase = infer_runtime_phase_from_decoder_layer_inputs(args, kwargs)
    set_runtime_phase(phase)
    # Mirror the summarized decode policy into the context so external
    # tooling can introspect the active policy of this forward call.
    set_runtime_decode_policy(decode_policy)
    return None


def install_phase_context_pre_hooks(network) -> None:
    """Install idempotent phase pre-hooks on all supported decoder layers.

    Public so downstream stacks (e.g. PLENA software) can re-install hooks
    after wrapping or re-building a quantized model.
    """

    decode_policy = _infer_runtime_decode_policy(network)
    if decode_policy is None:
        return

    hooked_layers = 0
    for module in network.modules():
        if not isinstance(module, _PHASE_HOOKED_DECODER_LAYERS):
            continue
        hooked_layers += 1
        if getattr(module, "_mase_phase_hook_installed", False):
            # A hook captures its decode policy at registration time, so an
            # unchanged policy means the existing hook is still correct.
            if module._mase_phase_hook_decode_policy == decode_policy:
                continue
            module._mase_phase_hook_handle.remove()
        handle = module.register_forward_pre_hook(
            partial(_phase_context_pre_hook, decode_policy=decode_policy),
            with_kwargs=True,
        )
        module._mase_phase_hook_installed = True
        module._mase_phase_hook_decode_policy = decode_policy
        module._mase_phase_hook_handle = handle

    if hooked_layers == 0:
        # Without hooks the runtime phase never leaves its "prefill" default,
        # so decode-side quantisation would silently stay inert.
        logging.warning(
            "Phase-aware quantized modules are present but no supported "
            "decoder layers were found to hook (%s). Decode-phase "
            "quantisation will NOT activate for this model family — extend "
            "_PHASE_HOOKED_DECODER_LAYERS or set the phase explicitly.",
            ", ".join(c.__name__ for c in _PHASE_HOOKED_DECODER_LAYERS),
        )


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
        config = config["config"]
        postfix = config.pop("name")
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
            quan_config = pass_args[n]

            quan_config = quan_config["config"]
            postfix = quan_config.pop("name")

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

        quan_config = pass_args[matched_pattern]["config"]
        postfix = quan_config["name"]

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

    Phase-split quantisation (prefill/decode) is expressed per module config
    via ``prefill`` / ``decode`` buckets — see
    ``chop.nn.quantized.modules.phase_config`` for the schema. A decode-only
    deployment (unquantised prefill chip, quantised decode chip) looks like:

    .. code-block:: python

        "model\\.layers\\.\\d+\\.self_attn\\.(q|k|v|o)_proj$": {
            "config": {
                "name": "mxint",
                "prefill": {"bypass": True},
                "decode": {
                    "weight_block_size": 32, "weight_width": 4,
                    "data_in_block_size": 32, "data_in_width": 8,
                },
            }
        }

    :return: The transformed torch.nn.Module.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.

    """
    # Defensive copy avoids mutating caller-owned pass_args (the pass pops
    # "name" out of nested selector configs), which matters for experiment
    # runners that reuse config dicts. Control blocks that may hold large
    # objects (calibration loaders, model handles) are carried by reference.
    _by_reference_keys = ("gptq", "rotation_search", "token_collector")
    pass_args = {
        key: (value if key in _by_reference_keys else deepcopy(value))
        for key, value in pass_args.items()
    }

    # If TOML has a [rotation_search] block, route the WHOLE quantize step
    # through the rotation search pass — it handles GPTQ, baseline module
    # replacement, and per-matmul rotate flag tuning end-to-end. Decisions
    # are cached to disk (default <gptq.checkpoint_dir>/rotation_decisions.json)
    # so a re-run skips the calib forwards entirely (mirrors GPTQ's
    # checkpoint resume).
    if "rotation_search" in pass_args:
        from .rotation_search import dispatch_rotation_search_block

        rot_cfg = pass_args.pop("rotation_search")
        return dispatch_rotation_search_block(network, pass_args, rot_cfg)

    # GPTQ pre-pass: quantize linear weights before module replacement.
    # With ``gptq_config["phase"] = "decode"`` the GPTQ result is stashed as
    # the decode weight bank while the FP weights are restored for prefill.
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
    # the phase-aware quantized modules rather than the original HF modules.
    install_phase_context_pre_hooks(network)

    return network, {}
