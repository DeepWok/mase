import logging
import math
import os
from pathlib import Path

from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, Optional, Tuple

import toml
import torch
import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

from chop.ir.common import MASE_IMPLICIT_FUNCS
from chop.nn import MASE_LEAF_LAYERS
from chop.nn.quantized import (
    quantized_func_map,
    quantized_module_map,
)

from transformers import PreTrainedModel
from transformers.utils.fx import symbolic_trace as hf_symbolic_trace
from transformers.utils.fx import HFTracer

logger = logging.getLogger(__name__)

# ----------------------------------------
#   Mase Tracer
# ----------------------------------------


class MaseTracer(fx.Tracer):
    def __init__(
        self,
        custom_leaf_modules: tuple[ModuleType] = (),
        custom_leaf_layers: tuple[torch.nn.Module] = (),
        custom_leaf_functions: tuple[Callable] = (),
        param_shapes_constant: bool = False,
    ) -> None:
        """Mase Tracer is an extended version of FX Tracer.

        :param custom_leaf_modules: Python modules whose functions should be wrapped automatically
            without needing to use fx.wrap(). Backward-compatibility for
            this parameter is guaranteed, defaults to ()
        :type custom_leaf_modules: tuple[ModuleType], optional
        :param custom_leaf_layers: Python functions that should be wrapped automatically without
            needing to use fx.wrap(). Backward compatibility for this
            parameter is guaranteed, defaults to ()
        :type custom_leaf_layers: tuple[torch.nn.Module], optional
        :param custom_leaf_functions: _description_, defaults to ()
        :type custom_leaf_functions: tuple[Callable], optional
        :param param_shapes_constant: When this flag is set,  calls to shape,
            size and a few other shape like attributes of a module's parameter
            will be evaluated directly, rather than returning a new Proxy value
            for an attribute access. Backward compatibility for this parameter
            is guaranteed, defaults to False
        :type param_shapes_constant: bool, optional
        """

        self.custom_leaf_layers = tuple(set(custom_leaf_layers))
        self.custom_leaf_modules = tuple(set(custom_leaf_modules))
        self.custom_leaf_functions = tuple(set(custom_leaf_functions))
        self.param_shapes_constant = param_shapes_constant
        super().__init__(
            self.custom_leaf_modules + (math,),
            self.custom_leaf_functions,
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


class MaseGraph:
    implicit_nodes = MASE_IMPLICIT_FUNCS

    def __init__(
        self,
        model,
        cf_args: Optional[Dict[str, Any]] = None,
        custom_ops: dict = None,
        hf_input_names: list = None,
    ) -> None:
        """Mase takes a torch.fx graph representation of a model and translates
        it into a customised representation (Mase graph IR). The Mase graph
        IR is a dataflow representation of the model with both software and
        hardware constraints.

        :param model: Input model to construct the MaseGraph. When a nn.Module is provided, this is parsed into a fx.GraphModule using the MaseTracer.
        :type model: torch.nn.Module | fx.GraphModule
        :param cf_args: _description_, defaults to None
        :type cf_args: Optional[Dict[str, Any]], optional
        """
        if isinstance(model, fx.GraphModule):
            self.model = model
            self.model.patched_op_names = []
            self.model.patched_custom_layers = []
            self.model.additional_inputs = []
        elif isinstance(model, torch.nn.Module):
            self.model = self.trace_torch_module(
                model,
                cf_args,
                custom_ops,
                hf_input_names=hf_input_names,
            )
        else:
            raise ValueError(
                f"Expected fx.GraphModule or nn.Module, but received model: {type(model)}"
            )

        self.cf_args = cf_args

    def trace_torch_module(
        self,
        model: torch.nn.Module,
        cf_args: Optional[Dict[str, Any]] = None,
        custom_ops: dict = None,
        hf_input_names: list = None,
    ):
        # * HuggingFace model
        if isinstance(model, PreTrainedModel):
            tracer_cls = HFTracer

            if custom_ops is not None:
                custom_modules = tuple(custom_ops.get("modules", {}).keys())
            else:
                custom_ops = {"modules": {}, "functions": {}}
                custom_modules = ()

            def wrap_is_leaf_module(hf_is_leaf_module):
                def is_leaf_module(
                    self, m: torch.nn.Module, module_qualified_name: str
                ) -> bool:
                    is_hf_built_in_leaf_module = hf_is_leaf_module(
                        self, m, module_qualified_name
                    )
                    is_custom_module = isinstance(m, custom_modules)
                    is_mase_leaf_layer = isinstance(m, MASE_LEAF_LAYERS)

                    return any(
                        (
                            is_hf_built_in_leaf_module,
                            is_custom_module,
                            is_mase_leaf_layer,
                        )
                    )

                return is_leaf_module

            setattr(
                tracer_cls,
                "is_leaf_module",
                wrap_is_leaf_module(tracer_cls.is_leaf_module),
            )

            graph_module = hf_symbolic_trace(
                model,
                tracer_cls=tracer_cls,
                input_names=hf_input_names,
            )
            graph_module.custom_ops = custom_ops

            # ! TO DO: remove this legacy stuff
            graph_module.patched_op_names = []
            graph_module.patched_custom_layers = []
            graph_module.additional_inputs = {}

        # * Other models
        else:
            # MASE internal auto-wrapped functions/layers
            custom_leaf_modules = ()
            custom_leaf_functions = ()
            custom_leaf_layers = ()
            # quantized functions/layers
            custom_leaf_functions += tuple(quantized_func_map.values())
            custom_leaf_layers += tuple(quantized_module_map.values())
            # patched functions/layers
            patched_nodes = getattr(model, "patched_nodes", None)
            if patched_nodes is not None:
                custom_leaf_modules += tuple(patched_nodes["modules"])
                custom_leaf_functions += tuple(patched_nodes["functions"])
                custom_leaf_layers += tuple(patched_nodes["layers"])

            self.tracer = MaseTracer(
                custom_leaf_modules=custom_leaf_modules,
                custom_leaf_functions=custom_leaf_functions,
                custom_leaf_layers=custom_leaf_layers,
            )

            graph_module = fx.GraphModule(model, self.tracer.trace(model, cf_args))

            if patched_nodes is not None:
                graph_module.patched_op_names = [
                    obj.__name__.lower()
                    for obj in model.patched_nodes["layers"]
                    + model.patched_nodes["functions"]
                ]
                # these are layers we believe the user will provide system verilog for
                graph_module.patched_custom_layers = model.patched_nodes["layers"]
                graph_module.additional_inputs = model.patched_nodes[
                    "additional_inputs"
                ]
                graph_module.patched_nodes = model.patched_nodes
            else:
                graph_module.patched_op_names = []
                graph_module.patched_custom_layers = []
                graph_module.additional_inputs = {}

        return graph_module

    @classmethod
    def from_module(
        cls,
        model: torch.nn.Module,
        cf_args: Optional[Dict[str, Any]] = None,
        custom_ops: dict = {},
    ):
        assert isinstance(
            model, torch.nn.Module
        ), f"model must be a torch.nn.Module. Received: {type(model)}"

        graph_module = self.trace_torch_module(model, cf_args, custom_ops)
        return cls(model=graph_module, cf_args=cf_args)

    def draw(self, file="mase_graph.svg"):
        drawer = FxGraphDrawer(self.model, "masegraph")
        drawer.get_dot_graph().write_svg(file)

    @property
    def fx_graph(self):
        return self.model.graph

    @fx_graph.setter
    def fx_graph(self, graph: fx.Graph):
        self.model.graph = graph

    @property
    def nodes(self):
        return self.model.graph.nodes

    @property
    def modules(self):
        return dict(self.model.named_modules())
