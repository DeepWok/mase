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
import onnx
import numpy as np

from pytorch_quantization import quant_modules, calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor

from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
from ....utils import deepcopy_mase_graph
from .utils import INT8Calibrator

def tensorrt_quantize_transform_pass(graph, pass_args=None):
    quantizer = Quantizer(pass_args)
    trt_graph_path = quantizer.pytorch_to_trt(graph)

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {'trt_graph_path': trt_graph_path}

class Quantizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def prepare_save_path(self, method: str):
        """Creates and returns a save path for the model."""
        root = Path(__file__).resolve().parents[7]
        current_date = datetime.now().strftime("%Y_%m_%d")
        save_dir = root / f"mase_output/TensorRT/Quantization/{method}" / current_date
        save_dir.mkdir(parents=True, exist_ok=True)

        existing_versions = len(os.listdir(save_dir))
        version = "version_0" if existing_versions==0 else f"version_{existing_versions}"

        save_dir = save_dir / version
        save_dir.mkdir(parents=True, exist_ok=True)

        return save_dir / f"model.{method.lower()}"
    
    def get_config(self, name: str):
        """Retrieve specific configuration from the instance's config dictionary or return default."""
        return self.config.get(name, 'default')
    
    def pre_quantization_test(self, model):
        """Evaluate pre-quantization performance."""
        print("Evaluate pre-quantization performance...")
        # Add evaluation code here

    def pytorch_quantize(self, graph):
        """Applies quantization procedures to PyTorch graph based on type."""
        # Add quantization code here

    def pytorch_to_trt(self, graph):
        """Converts PyTorch model to TensorRT format."""
        # Model is first converted to ONNX format and then to TensorRT
        ONNX_path = self.pytorch_to_ONNX(graph.model)
        self.summarize_ONNX_graph(ONNX_path)
        TRT_path = self.ONNX_to_TRT(ONNX_path)

        return TRT_path
        
    def ONNX_to_TRT(self, ONNX_path):
        self.logger.info("Converting PyTorch model to TensorRT...")
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
        config.max_workspace_size = 1 << 30
        #TODO add multiprecision exportation
        if self.config['default']['config']['precision'] == 'FP16':
            config.set_flag(trt.BuilderFlag.FP16)

        #TODO need to fix INT8 calibration
        elif self.config['default']['config']['precision'] == 'INT8':
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = INT8Calibrator(
                self.config['num_calibration_batches'], 
                self.config['data_module'].train_dataloader() 
                )

        else:
            Exception("Unsupported precision type. Please choose from 'FP16' or 'INT8'.")

        #TODO add optimizations based on input tensor
        # # Optimization profiles are needed for dynamic input shapes.
        # profile = builder.create_optimization_profile()
        # profile.set_shape("input_tensor_name", min=(1, 3, 224, 224), opt=(1, 3, 224, 224), max=(1, 3, 224, 224))  # Change based on model input.
        # config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)

        save_path = self.prepare_save_path(method='TRT')
        with open(save_path, "wb") as f:
            f.write(engine.serialize())

        self.logger.info(f"TensorRT Conversion Complete. Stored trt model to {save_path}")
        return save_path

    def pytorch_to_ONNX(self, model):
        """Converts PyTorch model to ONNX format and saves it."""
        self.logger.info("Converting PyTorch model to ONNX...")

        save_path = self.prepare_save_path(method='ONNX')

        dataloader = self.config['data_module'].train_dataloader()  
        train_sample = next(iter(dataloader))[0]
        train_sample = train_sample.to(self.config['accelerator'])

        torch.onnx.export(model, train_sample, save_path, export_params=True, opset_version=11, 
                          do_constant_folding=True, input_names=['input'])
        self.logger.info(f"ONNX Conversion Complete. Stored ONNX model to {save_path}")
        return save_path
    
    def summarize_ONNX_graph(self, onnx_model_path):
        # Load ONNX model
        model = onnx.load(onnx_model_path)
        graph = model.graph

        # Header for the table
        header = "Layer Name               | Type         | Input Shape(s)                       | Output Shape(s)"
        divider = "-" * len(header)

        # Start logging
        self.logger.info("\n" + divider + "\n" + header + "\n" + divider)

        # Iterate through each node (layer) in the graph
        for i, node in enumerate(graph.node):
            layer_name = node.name or f"Layer_{i}"  # Some nodes might not have a name
            layer_type = node.op_type

            # Retrieve input and output shapes
            input_shapes = [str(graph.value_info[input_name].type.tensor_type.shape) for input_name in node.input if input_name in graph.value_info]
            output_shapes = [str(graph.value_info[output_name].type.tensor_type.shape) for output_name in node.output if output_name in graph.value_info]

            # Format the shapes for better readability
            input_shapes_str = ', '.join(input_shapes) or 'Unknown'
            output_shapes_str = ', '.join(output_shapes) or 'Unknown'

            # Create the log entry for this layer
            log_entry = f"{layer_name:<25} | {layer_type:<12} | {input_shapes_str:<37} | {output_shapes_str}"
            
            # Log the entry
            self.logger.info(log_entry)

        self.logger.info(divider)  # End with a divider