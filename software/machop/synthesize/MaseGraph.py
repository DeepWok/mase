import logging
import toml
import pprint
import torch
import torch.fx
from torch import nn
from torch.fx import symbolic_trace

pp = pprint.PrettyPrinter(depth=4)


def get_module_by_name(model, request_name):
    for name, layer in model.named_modules():
        if name == request_name:
            return layer
    return None


# Mase takes a torch.fx graph representation of a model and translates
# it into a customised representation (Mase graph IR). The Mase graph
# IR is a dataflow representation of the model with both software and
# hardware constraints.
class MaseGraph:
    # Supported custom layer
    custom_layers = ["linear", "relu"]
    # Parameters for the custom layer
    hw_parameters = {
        nn.Linear: [
            "ACT_WIDTH",
            "W_WIDTH",
            "OUTPUT_WIDTH",
            "VECTOR_SIZE",
            "NUM_VECTORS",
            "PARALLELISM",
            "COMPUTE_TYPE",
        ],
        nn.ReLU: ["NUM", "ACT", "ACT_WIDTH"],
    }
    # Supported compilation targets
    compile_targets = ["HLS", "INTERNAL", "CUSTOM"]

    def __init__(
        self, model=None, save_name=None, emit=False, simulate=False, evaluate=False
    ):
        self.model = model
        self.save_name = save_name
        self.fx_graph = None
        self.parse()

    def parse(self):
        model = self.model
        # logging.debug(model)
        trace = torch.fx.symbolic_trace(model)
        trace.graph.lint()
        self.fx_graph = trace.graph
        logging.debug(self.fx_graph)
        self._add_hw_parameters()
        self._pre_synthesis_check()
        self.optimise()
        self.verify()

    def _pre_synthesis_check(self):
        count = {
            "placeholder": 0,
            "get_attr": 0,
            "call_function": 0,
            "call_method": 0,
            "call_module": 0,
            "output": 0,
        }
        layer_types = []
        for node in self.fx_graph.nodes:
            if node.op in count:
                count[node.op] += 1
            else:
                raise NotImplementedError(f"Unknown node op: {node_op!r}.")
        for node in self.fx_graph.nodes:
            if node.op == "call_module":
                node_layer = get_module_by_name(self.model, node.target)
                assert node_layer, "Cannot find {node.target} in the original module"
                if type(node_layer) not in layer_types:
                    layer_types.append(type(node_layer))
        logging.debug(
            f"""Network overview: 
{count}
Layer types: 
{layer_types}"""
        )

    # The hardware parameters are attached to each node in the form of metadata.
    # These parameters include the optimisation parameters for each layer. Also,
    # the parameter 'target' denotes which flow is used to generate the RTL code.
    # Target must be one of ['HLS', 'INTERNAL', 'CUSTOM'], where 'CUSTOM' is used
    # for used-provided IP (using modify-hw).
    def _add_hw_parameters(self):
        for node in self.fx_graph.nodes:
            node_layer = get_module_by_name(self.model, node.target)
            if type(node_layer) not in self.hw_parameters:
                logging.warning(
                    f"{node} is not found in the internal library and will be generated using HLS."
                )
                node.meta["target"] = "HLS"
            else:
                node.meta = dict.fromkeys(self.hw_parameters[type(node_layer)], None)
                node.meta["target"] = "INTERNAL"
        self._init_hw_parameters()

    def optimise(self):
        # Jianyi TODO
        return

    def verify(self):
        for node in self.fx_graph.nodes:
            # Each node must have a valid compilation target
            assert (
                node.meta["target"] in self.compile_targets
            ), "Unknown compilation target found: {}".format(node.meta["target"])
            for key, value in node.meta.items():
                assert value != None, f"Unspecified parameters {key} in: {node}"
        self._verify_hw_parameters()

    def _verify_hw_parameters(self):
        for name in self.custom_layers:
            replace_fn = getattr(self, f"_verify_parameters_{name}")
            replace_fn()

    def _verify_parameters_linear(self):
        target = nn.Linear
        for node in self.fx_graph.nodes:
            node_layer = get_module_by_name(self.model, node.target)
            if isinstance(node_layer, target):
                # Jianyi TODO
                ACT_WIDTH = node.meta["ACT_WIDTH"]
                assert (
                    ACT_WIDTH > 0
                ), f"Invalid hardware parameter ACT_WIDTH = {ACT_WIDTH}. {node}"
                W_WIDTH = node.meta["W_WIDTH"]
                assert (
                    W_WIDTH > 0
                ), f"Invalid hardware parameter W_WIDTH = {W_WIDTH}. {node}"
                OUTPUT_WIDTH = node.meta["OUTPUT_WIDTH"]
                assert (
                    OUTPUT_WIDTH > 0
                ), f"Invalid hardware parameter OUTPUT_WIDTH = {OUTPUT_WIDTH}. {node}"
                VECTOR_SIZE = node.meta["VECTOR_SIZE"]
                assert (
                    VECTOR_SIZE > 0
                ), f"Invalid hardware parameter VECTOR_SIZE = {VECTOR_SIZE}. {node}"
                NUM_VECTORS = node.meta["NUM_VECTORS"]
                assert (
                    NUM_VECTORS > 0
                ), f"Invalid hardware parameter NUM_VECTORS = {NUM_VECTORS}. {node}"
                PARALLELISM = node.meta["PARALLELISM"]
                assert (
                    PARALLELISM > 0
                ), f"Invalid hardware parameter PARALLELISM = {PARALLELISM}. {node}"
                COMPUTE_TYPE = node.meta["COMPUTE_TYPE"]
                assert COMPUTE_TYPE in [
                    "int",
                    "float",
                ], f"Invalid hardware parameter COMPUTE_TYPE = {COMPUTE_TYPE}. {node}"

    def _verify_parameters_relu(self):
        target = nn.ReLU
        for node in self.fx_graph.nodes:
            node_layer = get_module_by_name(self.model, node.target)
            if isinstance(node_layer, target):
                NUM = node.meta["NUM"]
                assert NUM > 0, f"Invalid hardware parameter NUM = {NUM}. {node}"
                ACT = node.meta["ACT"]
                assert ACT > 0, f"Invalid hardware parameter ACT = {ACT}. {node}"
                ACT_WIDTH = node.meta["ACT_WIDTH"]
                assert (
                    ACT_WIDTH > 0
                ), f"Invalid hardware parameter ACT_WIDTH = {ACT_WIDTH}. {node}"

    def _init_hw_parameters(self):
        for name in self.custom_layers:
            replace_fn = getattr(self, f"_init_parameters_{name}")
            replace_fn()

    def _init_parameters_linear(self):
        target = nn.Linear
        for node in self.fx_graph.nodes:
            node_layer = get_module_by_name(self.model, node.target)
            if isinstance(node_layer, target):
                # Jianyi TODO
                node.meta["ACT_WIDTH"] = 32
                node.meta["W_WIDTH"] = 32
                node.meta["OUTPUT_WIDTH"] = 32
                node.meta["VECTOR_SIZE"] = 1
                node.meta["NUM_VECTORS"] = 1
                node.meta["PARALLELISM"] = 1
                node.meta["COMPUTE_TYPE"] = "int"

    def _init_parameters_relu(self):
        target = nn.ReLU
        for node in self.fx_graph.nodes:
            node_layer = get_module_by_name(self.model, node.target)
            if isinstance(node_layer, target):
                # Jianyi TODO
                node.meta["NUM"] = 1
                node.meta["ACT"] = 1
                node.meta["ACT_WIDTH"] = 32
