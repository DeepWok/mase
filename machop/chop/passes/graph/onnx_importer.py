import torch
import torch.nn as nn
import torch.fx as fx
import onnx
from .onnx_utils import ONNX_OP_MAPPING


class ONNX_Importer:
    def __init__(self, onnx_graph):
        self.onnx_graph = onnx_graph
        self.fx_graph = fx.Graph()
        self.gm = fx.GraphModule(nn.Module(), self.fx_graph)
        self.fx_nodes = {}

        # Parse onnx constants and attributes
        self.onnx_constants = {
            onnx_node.name: onnx_node
            for onnx_node in onnx_graph.graph.node
            if onnx_node.op_type == "Constant"
        }

        self.onnx_attributes = {
            attr.name: attr for attr in onnx_graph.graph.initializer
        }

        # Obtain mapping of input edge to producing node
        # Initialize with graph inputs
        self.edge_mapping = {
            in_node.name: in_node for in_node in onnx_graph.graph.input
        }

        for node in onnx_graph.graph.node:
            for out in node.output:
                # Every edge label should be unique
                assert out not in self.edge_mapping, "Doubly defined edge label"
                self.edge_mapping[out] = node

    def raise_to_fx(self):
        self.check_supported_ops()

        # Register onnx graph attributes (initializers)
        for attr_path in self.onnx_attributes.keys():
            # Model attributes
            if attr_path.split(".")[0] == "model":
                module_str = ".".join(attr_path.split(".")[1:-1])
                attr_name = attr_path.split(".")[-1]
                self.init_submodule(self.gm, module_str)  # in case not yet initialized
                setattr(
                    self.gm.get_submodule(module_str),
                    attr_name,
                    torch.from_numpy(
                        onnx.numpy_helper.to_array(self.onnx_attributes[attr_path])
                    ),
                )

            # ONNX namespace
            else:
                # Set at top module level
                setattr(
                    self.gm,
                    attr_path,
                    torch.from_numpy(
                        onnx.numpy_helper.to_array(self.onnx_attributes[attr_path])
                    ),
                )

        # Create fx placeholders from ONNX graph inputs
        for in_node in self.onnx_graph.graph.input:
            name = self.clean_name(in_node.name)
            _ = self.fx_graph.create_node(op="placeholder", name=name, target=name)

        # Initialize all nodes
        for onnx_node in self.onnx_graph.graph.node:
            # Don't register Constant nodes
            if onnx_node.op_type == "Constant":
                continue

            name = self.clean_name(onnx_node.name)

            # Register submodule
            node_module = ".".join(name.split(".")[:-1])
            self.init_submodule(self.gm, node_module)  # in case not yet initialized

            new_node = self.fx_graph.create_node(
                op=ONNX_OP_MAPPING[onnx_node.op_type]["fx_op"],
                name=name,
                target=ONNX_OP_MAPPING[onnx_node.op_type]["target"],
            )
            self.fx_nodes[name] = new_node

        # Map arguments
        for onnx_node in self.onnx_graph.graph.node:
            fx_node_name = self.clean_name(onnx_node.name)
            for input_edge in onnx_node.input:
                # Input from model parameters mapped from onnx graph attributes
                if input_edge in [
                    attr.name for attr in self.onnx_graph.graph.initializer
                ]:
                    self.fx_nodes[fx_node_name].args += (input_edge,)

                # Input from onnx constant node
                elif self.edge_mapping[input_edge].name in self.onnx_constants.keys():
                    self.fx_nodes[fx_node_name].args += (
                        self.deserialize_constant(self.edge_mapping[input_edge]),
                    )

                # Input from other node
                elif input_edge in self.edge_mapping.keys():
                    self.fx_nodes[fx_node_name].args += (
                        self.get_node_by_name(
                            self.clean_name(self.edge_mapping[input_edge].name)
                        ),
                    )

                else:
                    raise RuntimeError(
                        f"Unrecognized input edge {input_edge} for ONNX node {onnx_node.name}."
                    )

        return self.gm

    def clean_name(self, name):
        node_name = name[1:] if (name[0] == "/") else name
        node_name = node_name.replace("/", "_").replace(".", "_").lower()
        return node_name

    def init_submodule(self, gm, target):
        try:
            # If submodule already exists, do not overwrite
            _ = gm.get_submodule(target)
        except:
            # If it doesn't, initialize submodule
            gm.add_submodule(target, nn.Module())

    def deserialize_constant(self, constant_node):
        assert len(constant_node.attribute) == 1
        tensor_proto = onnx.helper.get_attribute_value(constant_node.attribute[0])
        return torch.from_numpy(onnx.numpy_helper.to_array(tensor_proto))

    def get_node_by_name(self, name):
        for node in self.fx_graph.nodes:
            if node.name == name:
                return node
        raise RuntimeError(f"Node {name} not found")

    def check_supported_ops(self):
        ops = []
        unmapped_ops = []
        for node in self.onnx_graph.graph.node:
            if node.op_type not in ops:
                ops.append(node.op_type)
        for op in ops:
            if op not in ONNX_OP_MAPPING.keys():
                unmapped_ops.append(op)
        if "Constant" in unmapped_ops:
            unmapped_ops.remove("Constant")
        if len(unmapped_ops) > 0:
            raise RuntimeError(
                f"Found following unsupported ops in the obtained model: {unmapped_ops}"
            )
