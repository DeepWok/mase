import logging
import math

import torch
from torch.fx import GraphModule, Tracer
from torch.fx import wrap as fx_wrap
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerNorm

CUSTOM_LEAF_MODULES = [T5LayerNorm]
CUSTOM_LEAF_FUNCTIONS = []


def mark_as_leaf_module(cls: type):
    assert issubclass(cls, torch.nn.Module)
    if cls in CUSTOM_LEAF_MODULES:
        logging.warning(f"Class {cls} was already marked as leaf module")
    else:
        CUSTOM_LEAF_MODULES.append(cls)
    return cls


def mark_as_leaf_func(func):
    if func in CUSTOM_LEAF_FUNCTIONS:
        logging.warning(f"Function {func} was already marked as leaf function")
    else:
        CUSTOM_LEAF_FUNCTIONS.append(func)
    return func


# ----------------------------------------
# Currently tensor constructors cannot be traced.
# see https://pytorch.org/docs/stable/fx.html#miscellanea
# ----------------------------------------

MY_TENSOR_CONSTRUCTORS = []


def mark_as_tensor_constructor(func):
    if func in CUSTOM_LEAF_FUNCTIONS:
        logging.warning(f"Function {func} was already marked as leaf function")
    else:
        MY_TENSOR_CONSTRUCTORS.append(func)
    return func


@mark_as_tensor_constructor
def torch_zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


@mark_as_tensor_constructor
def torch_ones(*args, **kwargs):
    return torch.ones(*args, **kwargs)


@mark_as_tensor_constructor
def torch_arange(*args, **kwargs):
    return torch.arange(*args, **kwargs)


class MaseTracer(Tracer):
    def __init__(
        self,
        autowrap_modules=None,
        autowrap_functions=None,
        param_shapes_constant: bool = True,
    ) -> None:
        logging.debug(
            f"Current custom leaf functions: {CUSTOM_LEAF_FUNCTIONS+MY_TENSOR_CONSTRUCTORS+CUSTOM_LEAF_MODULES}\n"
            + "Current custom leaf models: {CUSTOM_LEAF_MODULES}"
        )
        if autowrap_modules is None:
            autowrap_modules = (math,)
        if autowrap_functions is None:
            autowrap_functions = tuple(CUSTOM_LEAF_FUNCTIONS + MY_TENSOR_CONSTRUCTORS)
        super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant)

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        is_tensor_constructor = is_custom_module = is_fx_built_in_module = False
        is_fx_built_in_module = super().is_leaf_module(m, module_qualified_name)
        if isinstance(m, tuple(CUSTOM_LEAF_MODULES)):
            is_custom_module = True
        if isinstance(m, type(MY_TENSOR_CONSTRUCTORS)):
            is_tensor_constructor = True
        return any((is_tensor_constructor, is_custom_module, is_fx_built_in_module))


def mase_symbolic_trace(root, concrete_args=None):
    tracer = MaseTracer()
    graph = tracer.trace(root, concrete_args=concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return GraphModule(tracer.root, graph, name)
