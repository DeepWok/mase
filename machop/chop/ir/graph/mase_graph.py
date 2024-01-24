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

from chop.passes.graph.common import MASE_IMPLICIT_FUNCS
from chop.passes.graph.transforms import utils as utils_passes
from chop.passes.graph.patching import MASE_LEAF_FUNCTIONS, MASE_LEAF_LAYERS
from chop.passes.graph.transforms.quantize import (
    quantized_func_map,
    quantized_module_map,
)
from torch.fx import wrap as fx_wrap

import onnx
from optimum.exporters.onnx import main_export
from ...tools.onnx_importer import ONNX_Importer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4].as_posix()

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


class MaseGraph:
    implicit_nodes = MASE_IMPLICIT_FUNCS

    def __init__(
        self,
        model: torch.nn.Module | str | onnx.onnx_ml_pb2.ModelProto,
        cf_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mase takes a torch.fx graph representation of a model and translates
        it into a customised representation (Mase graph IR). The Mase graph
        IR is a dataflow representation of the model with both software and
        hardware constraints.

        :param model: _description_
        :type model: torch.nn.Module | str | onnx.onnx_ml_pb2.ModelProto
        :param cf_args: _description_, defaults to None
        :type cf_args: Optional[Dict[str, Any]], optional
        """
        assert isinstance(
            model, (torch.nn.Module, str, onnx.onnx_ml_pb2.ModelProto)
        ), f"model must be a torch.nn.Module, checkpoint string or ONNX ModelProto. Received: {type(model)}"

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
            custom_leaf_layers += tuple(patched_nodes["layers"])
            custom_leaf_functions += tuple(patched_nodes["functions"])

        self.cf_args = cf_args

        # Defined model checkpoint, run optimum onnx export
        if isinstance(model, str):
            model_path = f"{ROOT}/mase_output/onnx/{model}/model.onnx"
            if not os.path.exists(model_path):
                main_export(
                    model,
                    output=f"{ROOT}/mase_output/onnx/{model}",
                    no_post_process=True,
                    model_kwargs={"output_attentions": True},
                )
            self.onnx_model = onnx.load(model_path)
            self.model = ONNX_Importer(self.onnx_model).raise_to_fx()

        # Defined ONNX ModelProto, raise to FX
        elif isinstance(model, onnx.onnx_ml_pb2.ModelProto):
            self.model = ONNX_Importer(model).raise_to_fx()

        # Defined torch.nn.Module, trace into fx graph
        else:
            # create graph module
            self.tracer = MaseTracer(
                custom_leaf_modules=custom_leaf_modules,
                custom_leaf_functions=custom_leaf_functions,
                custom_leaf_layers=custom_leaf_layers,
            )
            self.model = fx.GraphModule(model, self.tracer.trace(model, cf_args))
            if patched_nodes:
                self.model.patched_op_names = [
                    obj.__name__.lower()
                    for obj in model.patched_nodes["layers"]
                    + model.patched_nodes["functions"]
                ]
                # these are layers we believe the user will provide system verilog for
                self.model.patched_custom_layers = model.patched_nodes["layers"]
                self.model.additional_inputs = model.patched_nodes["additional_inputs"]
            else:
                self.model.patched_op_names = []
                self.model.patched_custom_layers = []
                self.model.additional_inputs = {}

    def draw(self, file="mase_graph.svg"):
        drawer = FxGraphDrawer(self.model, "masegraph")
        drawer.get_dot_graph().write_svg(file)

    @property
    def fx_graph(self):
        return self.model.graph

    @property
    def nodes(self):
        return self.model.graph.nodes

    @property
    def modules(self):
        return dict(self.model.named_modules())
