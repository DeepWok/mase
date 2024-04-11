import torch
import torch.nn as nn
import torch.fx as fx
import onnx
from .onnx_utils import ONNX_OP_MAPPING

import logging

class ONNX_Importer:
    def __init__(self, onnx_graph):
        self.onnx_graph = onnx_graph
        self.fx_graph = fx.Graph()
        self.gm = fx.GraphModule(nn.Module(), self.fx_graph)

        # fx.graph.nodes is not subscriptable, so we maintain this dict as new nodes are added
        self.fx_nodes = {}

        # Parse onnx constants and attributes
        self.onnx_constants = {
            onnx_node.name: onnx_node
            for onnx_node in onnx_graph.graph.node
            if onnx_node.op_type == "Constant"
        }

        # Weights and biases
        self.onnx_initializer_attributes = {
            attr.name: attr for attr in onnx_graph.graph.initializer
        }

        # Obtain mapping of input edge to producing node
        # Initialize with graph inputs
        self.edge_mapping = {
            in_node.name: in_node for in_node in onnx_graph.graph.input
        }

        unmapped_ops = []
        for node in onnx_graph.graph.node:
            if node.op_type != "Constant" and node.op_type not in ONNX_OP_MAPPING.keys():
                unmapped_ops.append(node.op_type)
                continue

            for out in node.output:
                # Every edge label should be unique
                assert out not in self.edge_mapping, "Doubly defined edge label"
                self.edge_mapping[out] = node

        if len(unmapped_ops) > 0:
            raise RuntimeError(
                f"Found following unsupported ops in the obtained model: {unmapped_ops}"
            )

    def raise_to_fx(self):
        # Register onnx graph attributes (initializers)
        for attr_path in self.onnx_initializer_attributes.keys():
            # Model attributes
            if attr_path.split(".")[0] == "model" or attr_path.split(".")[0] == "bert":
                module_str = ".".join(attr_path.split(".")[1:-1])
                attr_name = attr_path.split(".")[-1]
                self.init_submodule(self.gm, module_str)  # in case not yet initialized
                setattr(
                    self.gm.get_submodule(module_str),
                    attr_name,
                    torch.from_numpy(
                        onnx.numpy_helper.to_array(self.onnx_initializer_attributes[attr_path])
                    ),
                )

            # ONNX namespace
            else:
                # Set at top module level
                setattr(
                    self.gm,
                    attr_path,
                    torch.from_numpy(
                        onnx.numpy_helper.to_array(self.onnx_initializer_attributes[attr_path])
                    ),
                )

        # Create fx placeholders from ONNX graph inputs
        for in_node in self.onnx_graph.graph.input:
            name = self.clean_name(in_node.name)
            new_node = self.fx_graph.create_node(op="placeholder", name=name, target=name)
            self.fx_nodes[name] = new_node

        # Initialize all nodes
        for onnx_node in self.onnx_graph.graph.node:
            # Don't register Constant nodes
            if onnx_node.op_type == "Constant":
                continue

            name = self.clean_name(onnx_node.name)

            # Register submodule
            node_module = ".".join(onnx_node.name.split("/")[2:-1])
            self.init_submodule(self.gm, node_module)  # in case not yet initialized

            node_op = ONNX_OP_MAPPING[onnx_node.op_type]["fx_op"]
            node_target = ONNX_OP_MAPPING[onnx_node.op_type]["target"]
            new_node = self.fx_graph.create_node(
                op=node_op,
                name=name,
                target=node_target,
            )
            # print(f"Mapping node {name} with op {node_op} and target {node_target} to submodule {node_module}")
            self.fx_nodes[name] = new_node

        # Map arguments
        for onnx_node in self.onnx_graph.graph.node:
            # Don't register Constant nodes
            if onnx_node.op_type == "Constant":
                continue

            # TO DO: handle this
            if onnx_node.op_type == "Slice":
                continue

            # TO DO: handle this
            if onnx_node.op_type == "Transpose":
                continue

            fx_node_name = self.clean_name(onnx_node.name)

            # Map ONNX node inputs to fx node kwargs
            # ONNX node inputs can be represented as "inputs" or "attributes".

            # (1) First we map "inputs", which can come from initializer attributes (i.e. module parameters), ONNX constant
            #   or another node (through edge mapping)

            # Variadic inputs: input list of arbitrary length that maps to a single torch argument
            if len(onnx_node.input) > len(ONNX_OP_MAPPING[onnx_node.op_type]["input_mapping"]):
                # TO DO: it may be possible to have a variadic input + fixed length input
                assert len(ONNX_OP_MAPPING[onnx_node.op_type]["input_mapping"]) == 1

                torch_arg_name = ONNX_OP_MAPPING[onnx_node.op_type]["input_mapping"][0]
                self.fx_nodes[fx_node_name].kwargs = {
                        **self.fx_nodes[fx_node_name].kwargs,
                        torch_arg_name : onnx_node.input # then here
                    }
                continue

            for input_idx, input_name in enumerate(onnx_node.input):

                torch_arg_name = ONNX_OP_MAPPING[onnx_node.op_type]["input_mapping"][input_idx]

                # ONNX input is not mapped to a torch input, so skip
                if (torch_arg_name == ""):
                    continue

                # Input from model parameters mapped from onnx graph attributes
                if input_name in self.onnx_initializer_attributes.keys():
                    self.fx_nodes[fx_node_name].kwargs = {
                        **self.fx_nodes[fx_node_name].kwargs,
                        torch_arg_name : input_name
                    }

                # Input from onnx constant node
                elif self.edge_mapping[input_name].name in self.onnx_constants.keys():
                    self.fx_nodes[fx_node_name].kwargs = {
                        **self.fx_nodes[fx_node_name].kwargs,
                        torch_arg_name : self.deserialize_constant(self.edge_mapping[input_name])
                    }

                # Input from other node
                elif input_name in self.edge_mapping.keys():
                    # Call method nodes in an FX graph expect the first argument to be the parent node to call
                    #   the method on
                    if (ONNX_OP_MAPPING[onnx_node.op_type]["fx_op"] == "call_method" and input_idx == 0):
                        self.fx_nodes[fx_node_name].args += (self.fx_nodes[self.clean_name(self.edge_mapping[input_name].name)], )

                    else:
                        self.fx_nodes[fx_node_name].kwargs = {
                            **self.fx_nodes[fx_node_name].kwargs,
                            torch_arg_name : self.fx_nodes[self.clean_name(self.edge_mapping[input_name].name)]
                        }

                else:
                    raise RuntimeError(
                        f"Unrecognized input {input_name} for ONNX node {onnx_node.name}."
                    )

            # (2) Now map node attributes to kwargs
            for attribute_idx, attribute in enumerate(onnx_node.attribute):
                torch_arg_name = ONNX_OP_MAPPING[onnx_node.op_type]["attribute_mapping"][attribute_idx]

                # ONNX attribute is not mapped to a torch input, so skip
                if (torch_arg_name == ""):
                    continue

                if (ONNX_OP_MAPPING[onnx_node.op_type]["attribute_transform"][attribute_idx]):
                    attr = ONNX_OP_MAPPING[onnx_node.op_type]["attribute_transform"][attribute_idx](attribute.i)
                else:
                    attr = attribute.i

                self.fx_nodes[fx_node_name].kwargs = {
                    **self.fx_nodes[fx_node_name].kwargs,
                    torch_arg_name : attr
                }

            # print(f"Mapped arguments {self.fx_nodes[fx_node_name].args} to node {fx_node_name}")

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
