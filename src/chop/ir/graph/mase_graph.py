import logging
import math
import os
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, Optional, Tuple
import dill

import toml
import torch
import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer
import matplotlib.pyplot as plt

from transformers import PreTrainedModel
from transformers.utils.fx import symbolic_trace as hf_symbolic_trace
from transformers.utils.fx import HFTracer

from chop.ir.common import MASE_IMPLICIT_FUNCS
from chop.nn import MASE_LEAF_LAYERS
from chop.nn.quantized import (
    quantized_func_map,
    quantized_module_map,
)
from chop.tools import get_logger

from .mase_metadata import MaseMetadata

logger = get_logger(__name__)
logger.setLevel("INFO")

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


def trace_torch_module(
    model: torch.nn.Module,
    cf_args: Optional[Dict[str, Any]] = None,
    custom_ops: dict = None,
    hf_input_names: list = None,
):
    """
    Trace a torch.nn.Module using the MaseTracer. This function is a wrapper
    around the HFTracer and MaseTracer, and is used to trace a torch.nn.Module
    into a fx.GraphModule. The fx.GraphModule is a dataflow representation of
    the model with both software and hardware constraints. The MaseTracer is
    used to trace the model, and the custom_ops are used to provide custom
    operations to the tracer.

    Args:
        model (torch.nn.Module): Input model to trace.
        cf_args (Optional[Dict[str, Any]], optional): Concrete forward arguments to trace the model with. Defaults to None.
        custom_ops (dict, optional): Custom operations to be used in the model. Defaults to None.

    Returns:
        fx.GraphModule: Traced model as a fx.GraphModule.
    """
    # Forces contiguous memory layout for all parameters
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        state_dict[name] = param.contiguous()
    model.load_state_dict(state_dict)

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
                    self,
                    m,
                    module_qualified_name,
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

        tracer = MaseTracer(
            custom_leaf_modules=custom_leaf_modules,
            custom_leaf_functions=custom_leaf_functions,
            custom_leaf_layers=custom_leaf_layers,
        )

        graph_module = fx.GraphModule(model, tracer.trace(model, cf_args))

        if patched_nodes is not None:
            graph_module.patched_op_names = [
                obj.__name__.lower()
                for obj in model.patched_nodes["layers"]
                + model.patched_nodes["functions"]
            ]
            # these are layers we believe the user will provide system verilog for
            graph_module.patched_custom_layers = model.patched_nodes["layers"]
            graph_module.additional_inputs = model.patched_nodes["additional_inputs"]
            graph_module.patched_nodes = model.patched_nodes
        else:
            graph_module.patched_op_names = []
            graph_module.patched_custom_layers = []
            graph_module.additional_inputs = {}

    return graph_module


# ----------------------------------------
#   Mase Graph IR
# ----------------------------------------


class MaseGraph:
    implicit_nodes = MASE_IMPLICIT_FUNCS

    def __init__(
        self,
        model: torch.nn.Module | fx.GraphModule,
        cf_args: Optional[Dict[str, Any]] = None,
        custom_ops: dict = None,
        hf_input_names: list = None,
        skip_init_metadata: bool = False,
        add_metadata_args: dict = None,
    ) -> None:
        """MaseGraph is a dataflow representation of a model with both software and hardware constraints.
        The MaseGraph can be constructed from a torch.nn.Module:

        .. code-block:: python

            from chop.ir.graph import MaseGraph
            from transformers import BertModel

            model = BertModel.from_pretrained("bert-base-uncased")
            mase_graph = MaseGraph(model)

            # Or, equivalently:
            mase_graph = MaseGraph.from_module(model)


        A MaseGraph can also be constructed from a pre-traced fx.GraphModule:

        .. code-block:: python

            from chop.ir.graph import MaseGraph
            import torch
            import torch.fx as fx

            model = torch.nn.Linear(10, 10)
            traced_model = fx.symbolic_trace(model)
            mase_graph = MaseGraph(traced_model)

        A MaseGraph can be exported as follows:

        .. code-block:: python

            from chop.ir.graph import MaseGraph
            import torch
            import torch.fx as fx

            model = torch.nn.Linear(10, 10)
            traced_model = fx.symbolic_trace(model)
            mase_graph = MaseGraph(traced_model)
            mase_graph.export("masegraph")

        The MaseGraph can then be loaded from a checkpoint as follows:

        .. code-block:: python

            from chop.ir.graph import MaseGraph

            mase_graph = MaseGraph.from_checkpoint("masegraph")

        To visualize the MaseGraph, the `draw` method can be used:

        .. code-block:: python

            from chop.ir.graph import MaseGraph

            mase_graph = MaseGraph.from_module(model)
            mase_graph.draw("mase_graph.svg")

        Args:
            model (torch.nn.Module | fx.GraphModule): Input model to construct the MaseGraph.
            cf_args (Optional[Dict[str, Any]], optional): Concrete forward arguments to trace the model with. Defaults to None.
            custom_ops (dict, optional): Custom operations to be used in the model. Defaults to None.
            hf_input_names (list, optional): Input names for HuggingFace models. Defaults to None.
            skip_init_metadata (bool, optional): Skip initializing metadata for the nodes. Defaults to False.
            add_metadata_args (dict, optional): Additional arguments for metadata initialization. Defaults to None.

        Raises:
            ValueError: If the input model is not a torch.nn.Module or fx.Graph.
        """

        # is_huggingface flag is used in passes to automate dummy input generation etc
        if isinstance(model, PreTrainedModel):
            self.is_huggingface = True
        else:
            self.is_huggingface = False

        self.cf_args = cf_args

        # Generate the GraphModule according to the model type
        if isinstance(model, fx.GraphModule):
            self.model = model
            self.model.patched_op_names = []
            self.model.patched_custom_layers = []
            self.model.additional_inputs = []
        elif isinstance(model, torch.nn.Module):
            self.model = trace_torch_module(
                model,
                cf_args,
                custom_ops,
                hf_input_names=hf_input_names,
            )
        else:
            raise ValueError(
                f"Expected fx.GraphModule or nn.Module, but received model: {type(model)}"
            )

        # Initialize metadata for each node
        # todo: will need to move metadata analysis passes into chop.ir for this to work
        # if not skip_init_metadata and add_metadata_args is not None:
        #     mg, _ = passes.init_metadata_analysis_pass(self.fx_graph)
        #     mg, _ = passes.add_common_metadata_analysis_pass(
        #         mg,
        #         pass_args=add_metadata_args,
        #     )

    @classmethod
    def from_module(
        cls,
        model: torch.nn.Module,
        cf_args: Optional[Dict[str, Any]] = None,
        custom_ops: dict = {},
    ):
        """
        Construct a MaseGraph from a torch.nn.Module.

        Args:
            model (torch.nn.Module): Input model to construct the MaseGraph.
            cf_args (Optional[Dict[str, Any]], optional): Concrete forward arguments to trace the model with. Defaults to None.
            custom_ops (dict, optional): Custom operations to be used in the model. Defaults to {}.

        Returns:
            MaseGraph: Constructed MaseGraph.
        """
        assert isinstance(
            model, torch.nn.Module
        ), f"model must be a torch.nn.Module. Received: {type(model)}"

        graph_module = trace_torch_module(model, cf_args, custom_ops)
        return cls(
            model=graph_module,
            cf_args=cf_args,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        propagate_missing_metadata: bool = True,
    ):
        """
        Load a MaseGraph from a checkpoint. A MaseGraph checkpoint consists of two files:
        {checkpoint}.pt and {checkpoint}.mz. {checkpoint}.pt contains the GraphModule,
        and {checkpoint}.mz contains the MaseMetadata.

        If propagate_missing_metadata is set to True, the MaseGraph will attempt to propagate
        metadata for missing nodes. This is useful when the exported metadata is incomplete due
        to serialization errors.

        Args:
            checkpoint (str): Checkpoint to load the MaseGraph from.
            propagate_missing_metadata (bool, optional): Propagate metadata for missing nodes. Defaults to True.

        Returns:
            MaseGraph: Loaded MaseGraph.
        """
        with open(f"{checkpoint}.pt", "rb") as f:
            loaded_model = torch.load(f)

        assert isinstance(
            loaded_model, fx.GraphModule
        ), f"Expected fx.GraphModule, but received model: {type(loaded_model)}"

        mg = cls(loaded_model)

        with open(f"{checkpoint}.mz", "rb") as f:
            loaded_meta = dill.load(f)

        loaded_meta = {k: dill.loads(v) for k, v in loaded_meta.items()}

        for node in mg.nodes:
            if node.name in loaded_meta.keys():
                parameters = loaded_meta[node.name]
                node.meta["mase"] = MaseMetadata(
                    node=node,
                    model=loaded_model,
                )
                node.meta["mase"].parameters = parameters
            else:
                # todo: propagate metadata for missing nodes
                logger.warning(f"Node {node.name} not found in loaded metadata.")
                node.meta["mase"] = MaseMetadata(
                    node=node,
                    model=loaded_model,
                )

        for attr in [
            "class_for_deserialization",
            "config",
            "device",
        ]:
            if hasattr(mg.model, attr):
                setattr(mg, attr, getattr(mg.model, attr))

        return mg

    def export(
        self,
        fname: str = "masegraph",
    ):
        """
        Export the MaseGraph to a pair of files: {fname}.pt and {fname}.mz.
        {fname}.pt contains the GraphModule, and {fname}.mz contains the MaseMetadata.

        Args:
            fname (str): Filename to save the MaseGraph to. Defaults to "masegraph".
        """

        fname = fname.split(".")[0]
        logger.info(f"Exporting MaseGraph to {fname}.pt, {fname}.mz")

        logger.debug(f"Recompiling GraphModule to preserve any transforms...")
        self.model.recompile()

        # The following parameters must be set as attributes in the GraphModule
        # for tracing to work during deserialization. These get overwritten during
        # transform passes so they are read from the MaseGraph attributes (which are
        # set during the import process).
        logger.debug(f"Storing tracing parameters into mg.model for deserialization...")
        for attr in [
            "class_for_deserialization",
            "config",
            "device",
        ]:
            if hasattr(self, attr):
                logger.debug(f"Setting {attr}")
                setattr(self.model, attr, getattr(self, attr))

        logger.info(f"Exporting GraphModule to {fname}.pt")
        with open(f"{fname}.pt", "wb") as f:
            torch.save(self.model, f)

        logger.info(f"Exporting MaseMetadata to {fname}.mz")

        combined_meta = {}
        for node in self.nodes:
            parameters = node.meta["mase"].parameters
            try:
                pickled = dill.dumps(parameters)
                combined_meta[node.name] = pickled
            except Exception as e:
                logger.warning(f"Failed to pickle {node.op} node: {node.name}")
                logger.warning(e)

        with open(f"{fname}.mz", "wb") as f:
            dill.dump(combined_meta, f)

    def draw(self, file="mase_graph.svg"):
        """
        Draw the MaseGraph using the FxGraphDrawer.

        Args:
            file (str, optional): File to save the graph to. Defaults to "mase_graph.svg".
        """
        try:
            import pydot
        except:
            raise ImportError("pydot is required to draw the graph")
        drawer = FxGraphDrawer(self.model, "masegraph")
        dot_graph = drawer.get_dot_graph()
        # some dot_graph contains .weight and .bias, this cause pydot to crash in plotting
        # so we need to remove them
        # for instance, in BERT, you have bert_embeddings_word_embeddings.weight as a node,
        # this is not allowed in graphviz

        dot_string = dot_graph.to_string()
        dot_string = dot_string.replace(".weight", "_weight")
        dot_string = dot_string.replace(".bias", "_bias")
        # the following code snippet is how to plot in networkx, but it does not look nice
        # new_dot_graph = pydot.graph_from_dot_data(dot_string)
        # new_dot_graph = new_dot_graph[0]
        # graph = nx.drawing.nx_pydot.from_pydot(dot_graph)
        # nx.draw(graph)
        # plt.tight_layout()
        # plt.savefig("test.png", format="PNG")
        dot_graph = pydot.graph_from_dot_data(dot_string)
        dot_graph = dot_graph[0]
        dot_graph.write_svg(file)

    @property
    def fx_graph(self):
        """The fx.Graph representation of the MaseGraph.

        Returns:
            fx.Graph: fx.Graph representation of the MaseGraph.
        """
        return self.model.graph

    @property
    def nodes(self):
        """The nodes of the MaseGraph.

        Returns:
            list: List of nodes in the MaseGraph.
        """
        return self.model.graph.nodes

    @property
    def modules(self):
        """
        Get all the modules in the model.

        Returns:
            dict: Dictionary of all the modules in the model.
        """
        return dict(self.model.named_modules())

    @fx_graph.setter
    def fx_graph(self, graph: fx.Graph):
        self.model.graph = graph
