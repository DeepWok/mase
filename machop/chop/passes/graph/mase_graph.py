import logging
import os
import toml
import torch
import torch.fx as fx
from torch.fx import wrap as fx_wrap
from typing import Dict
import logging
import math

from chop.passes.transforms import utils as utils_passes
from chop.passes.common import MASE_IMPLICIT_FUNCS

logger = logging.getLogger(__name__)

# ----------------------------------------
#   Mase Tracer
# ----------------------------------------


class MaseTracer(fx.Tracer):
    """
    Mase Tracer is an extended version of FX Tracer.
    """

    def __init__(
        self,
        autowrap_modules=None,
        autowrap_functions=None,
        param_shapes_constant: bool = True,
    ) -> None:
        self.custom_leaf_modules = []
        self.custom_leaf_functions = []
        self.user_custom_leaf_modules = []
        self.user_custom_leaf_functions = []
        self.custom_tensor_constructors = []

        logger.debug(
            f"""Current custom leaf functions: {self.custom_leaf_functions}
                Current user-defined custom leaf functions: {self.custom_tensor_constructors+self.user_custom_leaf_functions}
                Current custom leaf modules: {self.custom_leaf_modules}
                Current user-defined custom leaf modules: {self.user_custom_leaf_modules}
                """
        )
        if autowrap_modules is None:
            autowrap_modules = (math,)
        if autowrap_functions is None:
            autowrap_functions = tuple(
                self.custom_leaf_functions
                + self.custom_tensor_constructors
                + self.user_custom_leaf_functions
            )
        super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant)

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        is_tensor_constructor = False
        is_custom_module = False
        is_fx_built_in_module = False
        is_user_leaf_module = False
        is_fx_built_in_module = super().is_leaf_module(m, module_qualified_name)
        if isinstance(m, tuple(self.custom_leaf_modules)):
            is_custom_module = True
        if isinstance(m, type(self.custom_tensor_constructors)):
            is_tensor_constructor = True
        if isinstance(m, tuple(self.user_custom_leaf_modules)):
            is_user_leaf_module = True
        return any(
            (
                is_tensor_constructor,
                is_custom_module,
                is_fx_built_in_module,
                is_user_leaf_module,
            )
        )

    def mark_as_leaf_module(self, cls: type):
        assert issubclass(cls, torch.nn.Module)
        if cls in self.custom_leaf_modules:
            logger.warning(f"Class {cls} was already marked as leaf module")
        else:
            self.custom_leaf_modules.append(cls)
        return cls

    def is_leaf_module_to_trace(self, cls: type) -> bool:
        return cls in self.custom_leaf_modules

    def mark_as_leaf_func(self, func):
        if func in self.custom_leaf_functions:
            logger.warning(f"Function {func} was already marked as leaf function")
        else:
            self.custom_leaf_functions.append(func)
        return func

    def mark_as_user_custom_leaf_module(self, cls: type):
        assert issubclass(cls, torch.nn.Module)
        if cls in self.user_custom_leaf_modules:
            logger.warning(f"Class {cls} was already marked as leaf module")
        else:
            self.user_custom_leaf_modules.append(cls)
        return cls

    def clear_user_custom_leaf_modules(self):
        self.user_custom_leaf_modules.clear()

    def mark_as_user_custom_leaf_function(self, func):
        if func in self.user_custom_leaf_functions:
            logger.warning(f"Function {func} was already marked as leaf function")
        else:
            self.user_custom_leaf_functions.append(func)
        return func

    def clear_user_custom_leaf_functions(self):
        self.user_custom_leaf_functions.clear()


def mase_symbolic_trace(root, concrete_args=None):
    tracer = MaseTracer()
    graph = tracer.trace(root, concrete_args=concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return fx.GraphModule(tracer.root, graph, name)


# ----------------------------------------
#   Mase Graph IR
# ----------------------------------------


"""
MaseGraph Passes
- > Analysis pass (add metadata), does not change graph

- > Transformation pass (add/remove nodes), changes graph
<tab> -> Quantization transformation + Analysis

- > Compose passes

PasssManager
-> certain transformation passes are not compatible with each other
-> Certain analysis passes are not compatible with each other
"""

"""
Mase takes a torch.fx graph representation of a model and translates
it into a customised representation (Mase graph IR). The Mase graph
IR is a dataflow representation of the model with both software and
hardware constraints.
"""


class MaseGraph:
    report = utils_passes.report
    verify = utils_passes.verify
    implicit_nodes = MASE_IMPLICIT_FUNCS

    def __init__(self, model=None, cf_args=None):
        """
        model = input model
        """
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(
                "Invalid model for generating a mase graph. Expected to be a valid nn.Module"
            )

        self.model = mase_symbolic_trace(model, cf_args)
        # self.tracer
        self.fx_graph = self.model.graph
        self.modules = dict(self.model.named_modules())
