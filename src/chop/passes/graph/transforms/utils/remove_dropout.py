from typing import Any, Dict, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.node import Argument, Target


def remove_dropout(
    graph_module: fx.GraphModule, fetch_module_by_target=None
) -> fx.GraphModule:
    """
    Removes all dropout layers from the module.
    """
    # fx_model = fx.symbolic_trace(model)

    class DropoutRemover(torch.fx.Transformer):
        def call_module(
            self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
        ) -> Any:
            if isinstance(self.submodules[target], nn.Dropout):
                assert len(args) == 1
                return args[0]
            else:
                return super().call_module(target, args, kwargs)

    graph_module = DropoutRemover(graph_module).transform()
    return graph_module
