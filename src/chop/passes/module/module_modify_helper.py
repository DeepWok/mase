import torch

from functools import reduce, partial
from copy import deepcopy
import logging
import inspect

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
    LlamaMLP,
    LlamaRMSNorm,
)

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
)

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeMLP,
)

from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
)

from chop.nn.quantized.modules.llada.modeling_llada import (
    LLaDALlamaBlock,
)

from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertSdpaSelfAttention,
)

roberta_prefix_map = {
    RobertaSdpaSelfAttention: "roberta_self_attention",
    RobertaSelfAttention: "roberta_self_attention",
    RobertaIntermediate: "roberta_intermediate",
    RobertaOutput: "roberta_output",
    RobertaSelfOutput: "roberta_self_output",
    RobertaClassificationHead: "roberta_classification_head",
}

llama_prefix_map = {
    LlamaAttention: "llama_self_attention",
    LlamaMLP: "llama_mlp",
    LlamaRMSNorm: "llama_rms_norm",
}

qwen3_prefix_map = {
    Qwen3Attention: "qwen3_self_attention",
    Qwen3MLP: "qwen3_mlp",
    Qwen3RMSNorm: "qwen3_rms_norm",
}

qwen3_moe_prefix_map = {
    Qwen3MoeAttention: "qwen3_moe_self_attention",
    Qwen3MoeMLP: "qwen3_moe_mlp",
}

gpt_oss_prefix_map = {
    GptOssAttention: "gpt_oss_self_attention",
}

llada_prefix_map = {
    LLaDALlamaBlock: "llada_block",
}

bert_prefix_map = {
    BertSelfAttention: "bert_self_attention",
    BertSdpaSelfAttention: "bert_self_attention",
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


def weight_replacement(x, y):
    target_state_dict = deepcopy(x.state_dict())
    missing_keys, unexpected_keys = y.load_state_dict(target_state_dict, strict=False)
    _restore_decode_fp_snapshot_if_available(x, y)
    if missing_keys:
        logging.warning(
            f"Missing keys when loading state_dict: {missing_keys} from {x} to {y}"
        )
    if unexpected_keys:
        logging.warning(
            f"Unexpected keys when loading state_dict: {unexpected_keys} from {x} to {y}"
        )
    return y


def _restore_decode_fp_snapshot_if_available(source_module, target_module):
    """Restore decode FP snapshot onto phase-aware linear targets.

    Why this hook exists:
    - GPTQ pre-pass rewrites `nn.Linear.weight` in-place before replacement.
    - In `fp_only` mode, decode needs preserved FP snapshots.
    - In `quantized` mode, we avoid keeping FP decode banks to save memory.
    """

    decode_weight = getattr(source_module, "_mase_decode_weight_fp", None)
    decode_bias = getattr(source_module, "_mase_decode_bias_fp", None)

    target_has_decode_weight = hasattr(target_module, "_decode_weight_fp")
    target_has_decode_bias = hasattr(target_module, "_decode_bias_fp")
    if not target_has_decode_weight:
        return

    decode_policy = getattr(target_module, "decode_policy", None)
    if decode_policy == "quantized":
        # Memory policy: quantized decode should not keep an additional FP
        # decode bank alive after replacement.
        target_module._decode_weight_fp = torch.empty(
            0, device=target_module.weight.device
        )
        if target_has_decode_bias:
            target_module._decode_bias_fp = torch.empty(
                0, device=target_module.weight.device
            )
    elif decode_policy == "fp_only":
        if decode_weight is not None:
            target_module._decode_weight_fp = decode_weight.to(
                device=target_module.weight.device,
                dtype=target_module.weight.dtype,
            )
        if (
            target_has_decode_bias
            and decode_bias is not None
            and getattr(target_module, "bias", None) is not None
        ):
            target_module._decode_bias_fp = decode_bias.to(
                device=target_module.weight.device,
                dtype=target_module.weight.dtype,
            )

    refresh_decode_runtime_bank = getattr(
        target_module, "refresh_decode_runtime_bank", None
    )
    if callable(refresh_decode_runtime_bank):
        refresh_decode_runtime_bank()


def get_module_by_name(network, name):
    return network.get_submodule(name)
    # names = name.split(sep='.')
    # return reduce(getattr, names, module)


def set_module_by_name(
    model, name, target_module, parent_name=None, current_name=None, parent_model=None
):
    if name == parent_name:
        setattr(parent_model, current_name, target_module)
        return model

    for n, module in model.named_children():
        ## compound module, go inside it
        new_parent_name = n if parent_name is None else f"{parent_name}.{n}"
        set_module_by_name(module, name, target_module, new_parent_name, n, model)
    return model


def replace_by_name(network, name, module):
    original = get_module_by_name(network, name)
    new = weight_replacement(original, module)
    network = set_module_by_name(network, name, new)
    return network


"""
instantiation of different supported modules
"""


def instantiate_linear(module, postfix, module_map, additional_module_args):
    linear_cls = module_map[f"linear_{postfix}"]
    has_bias = not (module.bias is None)
    orig_dtype = module.weight.dtype
    orig_device = module.weight.device

    # TODO: some transformed modules have "config" as an argument then extract the additional_module_args from it. Some directly take the additional_module_args.
    # Need to handle this better
    if "config" in inspect.signature(linear_cls.__init__).parameters:
        linear = linear_cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=has_bias,
            device=orig_device,
            dtype=orig_dtype,
            config=additional_module_args,
        )
    else:
        linear = linear_cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=has_bias,
            device=orig_device,
            dtype=orig_dtype,
            **additional_module_args,
        )

    return linear


def instantiate_conv2d(module, postfix, module_map, additional_module_args):
    conv2d_cls = module_map[f"conv2d_{postfix}"]
    has_bias = not (module.bias is None)
    # TODO: some transformed modules have "config" as an argument then extract the additional_module_args from it. Some directly take the additional_module_args.
    # Need to handle this better
    if "config" in inspect.signature(conv2d.__init__).parameters:
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


def instantiate_embedding(module, postfix, module_map, additional_module_args):
    embedding_cls = module_map[f"embedding_{postfix}"]
    if "config" in inspect.signature(embedding_cls.__init__).parameters:
        embedding = embedding_cls(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            config=additional_module_args,
        )
    else:
        embedding = embedding_cls(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            **additional_module_args,
        )
    return embedding


def instantiate_layernorm(module, postfix, module_map, additional_module_args):
    layernorm_cls = module_map[f"layernorm_{postfix}"]
    has_bias = not (module.bias is None)
    layernorm = layernorm_cls(
        normalized_shape=module.normalized_shape,
        eps=module.eps,
        elementwise_affine=module.elementwise_affine,
        bias=has_bias,
        **additional_module_args,
    )
    return layernorm


def instantiate_roberta_module(
    module, postfix, prefix, module_map, module_args, network_args
):
    roberta_cls = module_map[f"{prefix}_{postfix}"]

    roberta_module = roberta_cls(
        config=network_args,
        q_config=module_args,
    )
    return roberta_module


def instantiate_llama_module(
    module, postfix, prefix, module_map, module_args, network_args
):
    llama_cls = module_map[f"{prefix}_{postfix}"]

    llama_module = llama_cls(
        config=network_args,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        q_config=module_args,
    )
    # Keep replacement dtype/device aligned with the original HF module.
    # Why this is required:
    # - Llama quantized wrappers (RMS/MLP/Attention) can be constructed with
    #   default FP32 parameters.
    # - In mixed replacement flows, downstream quantized linear modules may be
    #   FP16/BF16, so FP32 activations from wrappers can trigger runtime dtype
    #   mismatches at F.linear boundaries.
    # - Aligning here makes replacement behavior consistent with instantiate_linear
    #   and avoids scattering ad-hoc casts in forward paths.
    ref_param = next(module.parameters(), None)
    if ref_param is not None:
        llama_module = llama_module.to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )
    return llama_module


def instantiate_qwen3_module(
    module, postfix, prefix, module_map, module_args, network_args
):
    qwen3_cls = module_map[f"{prefix}_{postfix}"]
    qwen3_module = qwen3_cls(
        config=network_args,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        q_config=module_args,
    )
    return qwen3_module


def instantiate_qwen3_moe_module(
    module, postfix, prefix, module_map, module_args, network_args
):
    cls = module_map[f"{prefix}_{postfix}"]
    kwargs = {
        "config": network_args,
        "q_config": module_args,
    }
    if hasattr(module, "layer_idx"):
        kwargs["layer_idx"] = module.layer_idx
    # Qwen3MoeMLP uses a custom intermediate_size per expert
    if hasattr(module, "intermediate_size"):
        kwargs["intermediate_size"] = module.intermediate_size
    return cls(**kwargs)


def instantiate_gpt_oss_module(
    module, postfix, prefix, module_map, module_args, network_args
):
    cls = module_map[f"{prefix}_{postfix}"]
    return cls(
        config=network_args,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        q_config=module_args,
    )


def instantiate_llada_module(
    module, postfix, prefix, module_map, module_args, network_args
):
    cls = module_map[f"{prefix}_{postfix}"]
    # LLaDALlamaBlock needs layer_id, config, cache, and q_config
    return cls(
        layer_id=module.layer_id,
        config=module.config,
        cache=module._LLaDABlock__cache,
        q_config=module_args,
    )


def instantiate_bert_module(
    module, postfix, prefix, module_map, module_args, network_args
):
    bert_cls = module_map[f"{prefix}_{postfix}"]

    bert_module = bert_cls(
        config=network_args,
        layer_idx=module.layer_idx,
        q_config=module_args,
    )
    return bert_module


def instantiate_module(module, postfix, module_map, additional_module_args):
    is_roberta, roberta_layer_name = check_module_instance(module, roberta_prefix_map)
    is_llama, llama_layer_name = check_module_instance(module, llama_prefix_map)
    is_qwen3_moe, qwen3_moe_layer_name = check_module_instance(
        module, qwen3_moe_prefix_map
    )
    is_qwen3, qwen3_layer_name = check_module_instance(module, qwen3_prefix_map)
    is_gpt_oss, gpt_oss_layer_name = check_module_instance(module, gpt_oss_prefix_map)
    is_llada, llada_layer_name = check_module_instance(module, llada_prefix_map)
    is_bert, bert_layer_name = check_module_instance(module, bert_prefix_map)

    module_args = additional_module_args["config"]
    network_args = additional_module_args.get("network_config", None)

    if isinstance(module, torch.nn.Linear):
        module = instantiate_linear(module, postfix, module_map, module_args)
    elif isinstance(module, torch.nn.Conv2d):
        module = instantiate_conv2d(module, postfix, module_map, module_args)
    elif isinstance(module, torch.nn.Embedding):
        module = instantiate_embedding(module, postfix, module_map, module_args)
    elif isinstance(module, torch.nn.LayerNorm):
        module = instantiate_layernorm(module, postfix, module_map, module_args)
    elif is_roberta:
        module = instantiate_roberta_module(
            module, postfix, roberta_layer_name, module_map, module_args, network_args
        )
    elif is_llama:
        module = instantiate_llama_module(
            module, postfix, llama_layer_name, module_map, module_args, network_args
        )
    elif is_qwen3_moe:
        module = instantiate_qwen3_moe_module(
            module, postfix, qwen3_moe_layer_name, module_map, module_args, network_args
        )
    elif is_qwen3:
        module = instantiate_qwen3_module(
            module, postfix, qwen3_layer_name, module_map, module_args, network_args
        )
    elif is_gpt_oss:
        module = instantiate_gpt_oss_module(
            module, postfix, gpt_oss_layer_name, module_map, module_args, network_args
        )
    elif is_llada:
        module = instantiate_llada_module(
            module, postfix, llada_layer_name, module_map, module_args, network_args
        )
    elif is_bert:
        module = instantiate_bert_module(
            module, postfix, llama_layer_name, module_map, module_args, network_args
        )
    else:
        raise ValueError(f"{module} is not supported.")
    return module


def manual_instantiate_module(module, module_name, module_map, additional_module_args):
    """
    manually replace a module with a new one that doesn't share the base class
    The additional_module_args MUST match the configuration argument of the new module
    Often use in ann2snn conversion. Converting activation module or quantizor module to neurons.
    """
    new_module = module_map[module_name](**additional_module_args["config"])
    return new_module
