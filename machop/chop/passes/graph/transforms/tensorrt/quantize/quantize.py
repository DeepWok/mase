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



class Quantizer:
    def __init__(self, config):
        self.config = config

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

    def pytorch_to_onnx(self, model):
        """Converts PyTorch model to ONNX format and saves it."""
        print("Converting PyTorch model to ONNX...")
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
        print(f"ONNX model saved to {save_path}")
        print("Conversion to ONNX done!")
        return save_path

    def calibrate_onnx(self, model, train_loader, onnx_path):
        """Calibrates ONNX model for quantization."""
        print("Calibrating model for quantization...") 
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 20  # Adjust size as needed
        engine = builder.build_engine(network, config)

        with open(str(onnx_path) + 'trt', 'wb') as f:
            f.write(engine.serialize())

        print("Calibration done!")

    def save_tensorrt_model(self, model, output_dir):
        """Saves the quantized TensorRT model."""
        print(f"Saving quantized TensorRT model in {output_dir}")
        # Add saving mechanism here

    # Add any additional methods here if necessary


def tensorrt_quantize_transform_pass(graph, pass_args=None):
    by = pass_args.pop("by")
    match by:
        case "type":
            graph = pytorch_quantize_by_type(graph, pass_args)
        case "name":
            graph = pytorch_quantize_by_name(graph, pass_args)
        case "regex_name":
            graph = pytorch_quantize_by_regex_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}
