import torch.nn as nn

from chop.ir import MaseGraph
from chop.tools import get_logger, deepgetattr
from chop.nn.modules.lora import LoRALinear

logger = get_logger(__name__)
logger.setLevel("DEBUG")


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
            setattr(mg.model, node.target, new_module)
            logger.info(f"Replaced {node.target} with LoRALinear module.")

    mg.model.recompile()

    return mg, {}
