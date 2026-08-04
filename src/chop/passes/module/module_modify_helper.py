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
    Qwen3DecoderLayer,
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
from chop.nn.quantized.modules.phase_config import (
    DECODE_FP_BIAS_ATTR,
    DECODE_FP_WEIGHT_ATTR,
    GPTQ_DECODE_WEIGHT_ATTR,
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
    Qwen3DecoderLayer: "qwen3_decoder_layer",
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

# Original module classes whose quantized replacement defines a `from_self`
# classmethod. These go through the generic `instantiate_from_self` helper,
# skipping family-specific instantiation logic.
from_self_prefix_map = {
    Qwen3DecoderLayer: "qwen3_decoder_layer",
    Qwen3Attention: "qwen3_self_attention",
    Qwen3RMSNorm: "qwen3_rms_norm",
    LlamaAttention: "llama_self_attention",
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
    target_state_dict = x.state_dict()
    missing_keys, unexpected_keys = y.load_state_dict(target_state_dict, strict=False)
    _transfer_phase_weight_banks(x, y)
    _restash_child_weight_banks(x, y)
    if missing_keys:
        logging.warning(
            f"Missing keys when loading state_dict: {missing_keys} from {x} to {y}"
        )
    if unexpected_keys:
        logging.warning(
            f"Unexpected keys when loading state_dict: {unexpected_keys} from {x} to {y}"
        )
    return y


def _transfer_phase_weight_banks(source_module, target_module):
    """Carry phase-split weight banks across the module-replacement seam.

    The GPTQ pre-pass runs on plain ``nn.Linear`` modules *before*
    replacement and leaves its results as attributes (they are not part of
    ``state_dict``), so they must be handed over explicitly:

    - ``_mase_gptq_weight_decode`` (``run_gptq(phase="decode")``): the GPTQ
      result becomes the target's decode weight bank while the state-dict
      weights (restored FP) stay as the prefill bank.
    - ``_mase_decode_weight_fp`` / ``_mase_decode_bias_fp``
      (``run_gptq(phase="prefill")``): the FP snapshot backs an ``fp_only``
      decode policy while the GPTQ weights serve prefill.
    """

    gptq_decode_weight = getattr(source_module, GPTQ_DECODE_WEIGHT_ATTR, None)
    if gptq_decode_weight is not None and hasattr(
        target_module, "adopt_decode_gptq_weight"
    ):
        target_module.adopt_decode_gptq_weight(gptq_decode_weight)

    fp_weight = getattr(source_module, DECODE_FP_WEIGHT_ATTR, None)
    if fp_weight is not None and hasattr(target_module, "adopt_decode_fp_snapshot"):
        if not getattr(target_module, "gptq", False) and not getattr(
            target_module, "bypass", False
        ):
            # The state-dict weights are already GPTQ-quantised; a prefill
            # bucket without gptq=True has just re-quantised them (RTN on
            # top of GPTQ), which is almost never intended.
            logging.warning(
                "GPTQ(phase='prefill') weights were re-quantised during "
                "replacement of %s — set 'gptq': True in the prefill bucket.",
                type(target_module).__name__,
            )
        target_module.adopt_decode_fp_snapshot(
            fp_weight, getattr(source_module, DECODE_FP_BIAS_ATTR, None)
        )


def _restash_child_weight_banks(source_module, target_module):
    """Carry per-child phase-bank stashes across a WHOLESALE module replacement.

    When a container (e.g. LlamaMLP) is replaced, its fresh children are plain
    ``nn.Linear``s: ``load_state_dict`` restores their weights but the GPTQ /
    FP-snapshot stash *attributes* on the old children are not state_dict
    entries and would be silently lost — the later per-linear replacement would
    then fall back to FP decode weights. Re-stash them on the same-named child
    so ``_transfer_phase_weight_banks`` finds them at that replacement.
    """

    stash_attrs = (GPTQ_DECODE_WEIGHT_ATTR, DECODE_FP_WEIGHT_ATTR, DECODE_FP_BIAS_ATTR)
    target_children = dict(target_module.named_modules())
    for name, src_child in source_module.named_modules():
        if not name or name not in target_children:
            continue
        tgt_child = target_children[name]
        if tgt_child is src_child:
            continue
        for attr in stash_attrs:
            val = getattr(src_child, attr, None)
            if val is not None and getattr(tgt_child, attr, None) is None:
                setattr(tgt_child, attr, val)


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


def instantiate_from_self(
    module, postfix, prefix, module_map, module_args, network_args
):
    """Build the quantized replacement via the class's own `from_self` classmethod.

    Quantized modules that know how to reconstruct themselves from the original
    (matching dtype/device and copying state_dict) expose a `from_self`
    classmethod. This helper just dispatches to it — no type-specific logic.
    """
    cls = module_map[f"{prefix}_{postfix}"]
    return cls.from_self(module, q_config=module_args)

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
    has_from_self, from_self_layer_name = check_module_instance(
        module, from_self_prefix_map
    )

    module_args = additional_module_args["config"]
    network_args = additional_module_args.get("network_config", None)

    if has_from_self:
        module = instantiate_from_self(
            module, postfix, from_self_layer_name, module_map, module_args, network_args
        )
    elif isinstance(module, torch.nn.Linear):
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
    elif is_qwen3:
        # Same constructor shape as the Llama modules:
        # cls(config=..., layer_idx=..., q_config=...)
        module = instantiate_llama_module(
            module, postfix, qwen3_layer_name, module_map, module_args, network_args
        )
    elif is_qwen3_moe:
        module = instantiate_qwen3_moe_module(
            module, postfix, qwen3_moe_layer_name, module_map, module_args, network_args
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
