import torch

from functools import reduce, partial
from copy import deepcopy
import logging
import inspect

try:
    from transformers.models.roberta.modeling_roberta import (
        RobertaSelfAttention,
        RobertaSdpaSelfAttention,
        RobertaClassificationHead,
        RobertaIntermediate,
        RobertaOutput,
        RobertaSelfOutput,
    )
except ImportError:
    # RobertaSdpaSelfAttention was removed in newer transformers.
    # Roberta quantization paths won't be available, but other models still work.
    from transformers.models.roberta.modeling_roberta import (
        RobertaSelfAttention,
        RobertaClassificationHead,
        RobertaIntermediate,
        RobertaOutput,
        RobertaSelfOutput,
    )
    RobertaSdpaSelfAttention = None

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

from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeAttention,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
)

from chop.nn.quantized.modules.llada.modeling_llada import (
    LLaDALlamaBlock,
)

try:
    from transformers.models.bert.modeling_bert import (
        BertSelfAttention,
        BertSdpaSelfAttention,
    )
except ImportError:
    from transformers.models.bert.modeling_bert import BertSelfAttention
    BertSdpaSelfAttention = None

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

glm4_moe_prefix_map = {
    Glm4MoeAttention: "glm4_moe_self_attention",
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
        if cls is None:
            continue  # Skip classes that weren't importable (e.g. RobertaSdpaSelfAttention in newer transformers)
        if isinstance(module, cls):
            return True, name
    return False, None


def weight_replacement(x, y):
    target_state_dict = deepcopy(x.state_dict())
    missing_keys, unexpected_keys = y.load_state_dict(target_state_dict, strict=False)
    if missing_keys:
        logging.warning(
            f"Missing keys when loading state_dict: {missing_keys} from {x} to {y}"
        )
    if unexpected_keys:
        logging.warning(
            f"Unexpected keys when loading state_dict: {unexpected_keys} from {x} to {y}"
        )
    return y


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
    # Freshly-constructed sub-modules (e.g. Qwen3RMSNorm's q_norm/k_norm
    # weights) default to fp32; without this, `fp32_weight * bf16_input`
    # promotes activations to fp32 and breaks downstream matmuls with bf16
    # KV tensors. Match the original module's dtype/device and copy weights.
    params = list(module.parameters())
    if params:
        device, dtype = params[0].device, params[0].dtype
        qwen3_module = qwen3_module.to(dtype=dtype, device=device)
        qwen3_module.load_state_dict(module.state_dict(), strict=True)
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


def instantiate_glm4_moe_module(
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
    is_glm4_moe, glm4_moe_layer_name = check_module_instance(
        module, glm4_moe_prefix_map
    )
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
    elif is_glm4_moe:
        module = instantiate_glm4_moe_module(
            module, postfix, glm4_moe_layer_name, module_map, module_args, network_args
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
