from copy import copy, deepcopy
import logging
import torch
import os
import sys
from pathlib import Path
from datetime import datetime
import tensorrt as trt
from pathlib import Path
from datetime import datetime

from pytorch_quantization import quant_modules, calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor

from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
from ....utils import deepcopy_mase_graph


def tensorrt_quantize_transform_pass(graph, pass_args=None):
    quantizer = Quantizer(pass_args)
    by = pass_args["by"]
    match by:
        case "type":
            trt_graph = quantizer.pytorch_to_trt(graph)
        case "name":
            ...
        case "regex_name":
            ...
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}


class Quantizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_config(self, name: str):
        """Retrieve specific configuration from the instance's config dictionary or return default."""
        return self.config.get(name, self.config['default'])['config']
    
    def pre_quantization_test(self, model):
        """Evaluate pre-quantization performance."""
        print("Evaluate pre-quantization performance...")
        # Add evaluation code here

    def pytorch_quantize(self, graph):
        """Applies quantization procedures to PyTorch graph based on type."""
        # Add quantization code here

    def pytorch_to_trt(self, graph):
        """Converts PyTorch model to TensorRT format."""
        self.logger.info("Converting PyTorch model to TensorRT...")

        # Converts and saves to path
        ONNX_path = self.pytorch_to_ONNX(graph.model)
        TRT_path = self.ONNX_to_TRT(ONNX_path)
        
    def ONNX_to_TRT(self, ONNX_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(ONNX_path, "rb") as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # Adjust workspace size as necessary.
        config.set_flag(trt.BuilderFlag.FP16)

        # Optimization profiles are needed for dynamic input shapes.
        profile = builder.create_optimization_profile()
        profile.set_shape("input_tensor_name", min=(1, 3, 224, 224), opt=(1, 3, 224, 224), max=(1, 3, 224, 224))  # Change based on model input.
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)

        with open("model.trt", "wb") as f:
            f.write(engine.serialize())

    def pytorch_to_ONNX(self, model):
        """Converts PyTorch model to ONNX format and saves it."""
        self.logger.info("Converting PyTorch model to ONNX...")
        # Prepare the save path
        root = Path(__file__).resolve().parents[7]
        current_date = datetime.now().strftime("%Y_%m_%d")
        save_dir = root / "mase_output/TensorRT/Quantization/ONNX" / current_date
        save_dir.mkdir(parents=True, exist_ok=True)

        existing_versions = [int(d.name.split("_")[-1]) for d in save_dir.parent.iterdir() if d.is_dir() and d.name.startswith(current_date)]
        version = "version_0" if not existing_versions else f"version_{max(existing_versions) + 1}"

        save_dir = save_dir / version
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "model.onnx"

        dataloader = self.config['train_generator'].dataloader    
        train_sample = next(iter(dataloader))[0]
        train_sample = train_sample.to(self.config['device'])

        torch.onnx.export(model, train_sample, save_path, export_params=True, opset_version=11, 
                          do_constant_folding=True, input_names=['input'])
        self.logger.info(f"ONNX Conversion Complete. Stored ONNX model saved to {save_path}")
        return save_path
