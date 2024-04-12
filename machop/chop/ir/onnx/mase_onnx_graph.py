import onnx
from optimum.exporters.onnx import main_export

import os

HOME = os.environ["HOME"]

from .utils import ONNX_OP_MAPPING


class MaseOnnxGraph:

    def __init__(
        self,
        model_proto: onnx.onnx_ml_pb2.ModelProto,
    ):
        self.graph = model_proto.graph

        # * Initializer attributes contain weights and biases
        # i.e. module parameters in Pytorch
        self.initializer_attributes = {
            attr.name: attr for attr in self.graph.initializer
        }

        # * Iterate through all nodes in the graph
        self.constants = {}
        self.edge_mapping = {
            in_node.name: in_node for in_node in self.graph.input
        }  # ? might not be necessary
        unmapped_ops = []

        for node in self.graph.node:
            # Set constants
            if node.op_type == "Constant":
                self.constants[node.name] = node

            else:
                # Check for unsupported ops
                if node.op_type not in ONNX_OP_MAPPING.keys():
                    unmapped_ops.append(node.op_type)
                    continue

            # Obtain mapping of input edge to producing node
            for out in node.output:
                # Every edge label should be unique
                assert out not in self.edge_mapping, "Doubly defined edge label"
                self.edge_mapping[out] = node

        if len(unmapped_ops) > 0:
            raise RuntimeError(
                f"Found following unsupported ops in the obtained model: {unmapped_ops}"
            )

    @classmethod
    def from_pretrained(cls, pretrained: str):
        model_path = f"{HOME}/.mase/onnx/{pretrained}/model.onnx"
        if not os.path.exists(model_path):
            main_export(
                pretrained,
                output=f"{HOME}/.mase/onnx/{pretrained}",
                no_post_process=True,
                model_kwargs={"output_attentions": True},
            )
        return cls(model_proto=onnx.load(model_path))
