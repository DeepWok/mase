import torch
import torch.nn as nn

from chop.ir import MaseGraph
from chop.tools import get_logger, deepgetattr, deepsetattr
from chop.nn.modules.lora import LoRALinear

logger = get_logger(__name__)
logger.setLevel("INFO")


def insert_lora_adapter_transform_pass(
    mg: MaseGraph,
    pass_args={},
):

    rank = pass_args.get("rank", 0)
    lora_alpha = pass_args.get("lora_alpha", 0.5)
    lora_dropout = pass_args.get("lora_dropout", 0.0)

    for node in mg.nodes:
        target = (
            deepgetattr(mg.model, node.target) if node.op == "call_module" else None
        )
        if node.op == "call_module" and isinstance(target, nn.Linear):
            new_module = LoRALinear(
                target.in_features,
                target.out_features,
                rank=rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            new_module.linear.weight = target.weight
            deepsetattr(mg.model, node.target, new_module)
            logger.info(
                f"Replaced node: {node.name}, target: {node.target} with LoRALinear module."
            )

    mg.model.recompile()

    return mg, {}


def fuse_lora_weights_transform_pass(
    mg: MaseGraph,
    pass_args={},
):
    for node in mg.nodes:
        target = (
            deepgetattr(mg.model, node.target) if node.op == "call_module" else None
        )
        if node.op == "call_module" and isinstance(target, LoRALinear):
            old_weights = target.linear.weight
            lora_a = target.lora_a.weight
            lora_b = target.lora_b.weight

            logger.info(f"Fusing LoRALinear weights for {node.target}.")
            logger.debug(f"Old weights: {old_weights.shape}")
            logger.debug(f"Lora A weights: {lora_a.shape}")
            logger.debug(f"Lora B weights: {lora_b.shape}")

            new_weights = old_weights + (target.alpha / target.rank) * lora_b @ lora_a
            logger.debug(f"New weights: {new_weights.shape}")

            new_linear = nn.Linear(
                target.linear.in_features,
                target.linear.out_features,
                bias=target.linear.bias,
            )
            new_linear.weight = nn.Parameter(new_weights)

            deepsetattr(mg.model, node.target, new_linear)

    return mg, {}
