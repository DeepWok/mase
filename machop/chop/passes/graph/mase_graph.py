import logging
import math
import os
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, Optional, Tuple

import toml
import torch
import torch.fx as fx
from chop.passes.common import MASE_IMPLICIT_FUNCS
from chop.passes.transforms import utils as utils_passes
from chop.passes.patching import MASE_LEAF_FUNCTIONS, MASE_LEAF_LAYERS
from torch.fx import wrap as fx_wrap

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
        custom_leaf_modules: tuple[ModuleType] = (),
        custom_leaf_layers: tuple[torch.nn.Module] = (),
        custom_leaf_functions: tuple[Callable] = (),
        param_shapes_constant: bool = False,
    ) -> None:
        """
        Construct a Tracer object.

        Args:
            custom_leaf_modules (Tuple[ModuleType]): defaults to `()`,
            Python modules whose functions should be wrapped automatically
            without needing to use fx.wrap(). Backward-compatibility for
            this parameter is guaranteed.

            custom_leaf_functions (Tuple[Callable, ...]): defaults to `()`,
            Python functions that should be wrapped automatically without
            needing to use fx.wrap(). Backward compatibility for this
            parameter is guaranteed.

            param_shapes_constant (bool): When this flag is set,  calls to shape,
            size and a few other shape like attributes of a module's parameter
            will be evaluated directly, rather than returning a new Proxy value
            for an attribute access. Backward compatibility for this parameter
            is guaranteed.
        """
        self.custom_leaf_layers = tuple(set(custom_leaf_layers))
        self.custom_leaf_modules = tuple(set(custom_leaf_modules))
        self.custom_leaf_functions = tuple(set(custom_leaf_functions))
        self.param_shapes_constant = param_shapes_constant
        super().__init__(
            self.custom_leaf_modules + (math,),
            self.custom_leaf_functions + MASE_LEAF_FUNCTIONS,
            self.param_shapes_constant,
        )

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        is_fx_built_in_leaf_module = super().is_leaf_module(m, module_qualified_name)
        is_mase_leaf_layers = isinstance(m, MASE_LEAF_LAYERS)
        is_custom_layer = isinstance(m, self.custom_leaf_layers)
        return any(
            (
                is_fx_built_in_leaf_module,
                is_mase_leaf_layers,
                is_custom_layer,
            )
        )


# ----------------------------------------
#   Mase Graph IR
# ----------------------------------------
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

    def __init__(
        self,
        model: torch.nn.Module = None,
        cf_args: Optional[Dict[str, Any]] = None,
        load_name: Optional[str] = None,
    ) -> None:
        if model is not None and load_name is not None:
            raise ValueError("model and load_name cannot be set at the same time.")
        elif model is not None:
            assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
        else:
            assert isinstance(load_name, str), "load_name must be a mase checkpoint"

        # create graph module
        if model is not None:
            patched = getattr(model, "patched", None)
            if patched is None:
                self.tracer = MaseTracer()
            else:
                self.tracer = MaseTracer(
                    custom_leaf_modules=tuple(model.patched_nodes["modules"]),
                    custom_leaf_layers=tuple(model.patched_nodes["layers"]),
                    custom_leaf_functions=tuple(model.patched_nodes["functions"]),
                )
            self.cf_args = cf_args
            self.model = fx.GraphModule(model, self.tracer.trace(model, cf_args))
            if patched:
                self.model.patched_op_names = model.patched_nodes["names"]
                # these are layers we believe the user will provide system verilog for
                self.model.patched_custom_layers = model.patched_nodes["layers"]
                self.model.additional_inputs = model.patched_nodes["additional_inputs"]
            else:
                self.model.patched_op_names = []
                self.model.patched_custom_layers = []
                self.model.additional_inputs = {}
        else:
            self.tracer = None
            self.cf_args = None
            self.model = torch.load(load_name)

    @property
    def fx_graph(self):
        return self.model.graph

    @property
    def modules(self):
        return dict(self.model.named_modules())
