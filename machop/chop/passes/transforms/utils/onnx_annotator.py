# A pass to convert a MASE graph to ONNX and annotate the relevant layers with sparsity
# information. The code here is derived from Zhewen's SparseCNN codebase:
# https://github.com/Yu-Zhewen/sparseCNN/blob/main/onnx_sparsity_attribute.py

from collections import OrderedDict
import torch.nn as nn
import torch
import onnx
import toml
from onnx import ModelProto, NodeProto
from pathlib import Path
from chop.tools.logger import getLogger


# Housekeeping -------------------------------------------------------------------------
logger = getLogger(__file__)
logger.propagate = False  # Avoid duplicate logs


def onnx_annotate_transform_pass(graph, **kwargs):
    # Export the model to ONNX
    path = kwargs["save_path"] / "dense.onnx"
    _torch_onnx_exporter(graph.model, torch.randn(1, 3, 224, 224), path)

    # Load the model back and annotate the relevant layers with sparsity attributes
    onnx_model = onnx.load(path)
    data_path = kwargs["data_path"]
    # NOTE: For now, these are hard-coded. Later we may want to make these configurable.
    # ww = weight width, dw = data width, aw = accumulator width
    _annotate_quantisation(onnx_model, ww=16, dw=16, aw=32, bfp=False)
    assert data_path.endswith(".toml"), "Only TOML files are supported for now"
    _annotate_sparsity_from_toml(onnx_model, data_path)
    onnx.save(onnx_model, kwargs["save_path"] / "sparse.onnx")

    return graph


# https://github.com/Xilinx/finn-base/blob/dev/src/finn/custom_op/base.py
def _set_nodeattr(node: NodeProto, name: str, value):
    # NOTE We do not check if the attribute already exists.
    attr = onnx.helper.make_attribute(name, value)
    node.attribute.append(attr)


def _annotate_quantisation(model: ModelProto, ww: int, dw: int, aw: int, bfp: bool):
    for node in model.graph.node:
        if node.op_type in ["Conv", "Gemm"]:
            _set_nodeattr(node, "weight_width", ww)
            _set_nodeattr(node, "acc_width", aw)
            _set_nodeattr(node, "block_floating_point", bfp)
        _set_nodeattr(node, "data_width", dw)
    logger.info("Quantisation annotation complete.")


def _annotate_sparsity_from_toml(model: onnx.ModelProto, path: Path):
    with open(path) as f:
        data = toml.load(f)

    # NOTE: Here, we're solely relying on the coherence of the TOML file and the ONNX
    # graph in terms of the order of the layers. This works for our use case.
    iterator = iter(data.items())
    for node in model.graph.node:
        if node.op_type == "Conv":
            info = next(iterator)
            sparsity_data = info[1]["avg"]
            logger.info(f"Annotating {node.name} - {info[0]}")
            _set_nodeattr(node, "input sparsity", sparsity_data)
    logger.info("Layer sparsity annotation complete.")


def _torch_onnx_exporter(model: nn.Module, input: torch.Tensor, path: Path):
    # We need to replace certain layers in the model for compatibility reasons
    replace_dict = {}
    for module in model.modules():
        # TODO: We need to add a clip node. This is a temporary fix. There may be other
        # unsupported layers as well.
        if isinstance(module, nn.ReLU6):
            replace_dict[module] = nn.ReLU()
    _replace_modules(model, replace_dict)

    # Export the model to the specified path in ONNX format. :)
    kwargs = {"verbose": False, "keep_initializers_as_inputs": True}
    torch.onnx.export(model, input, path, **kwargs)


# Substitutes certain layers in the model with their chosen replacements
def _replace_modules(model: nn.Module, replace_dict: dict):
    for module in model.modules():
        for name, submodule in module.named_children():
            if submodule in replace_dict.keys():
                new_submodule = replace_dict[submodule]
                assert hasattr(module, name)
                setattr(module, name, new_submodule)
