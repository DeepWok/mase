import logging
import torch
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import onnx
import onnxruntime as ort

def onnx_runtime_transform_pass(graph, pass_args="None"):
    onnx_runtime_session = ONNXRuntime()

    pytorch_model = graph.model

    do_test = pass_args['do_test']

    if do_test == 'before' or do_test == 'both':
        onnx_runtime_session.test_performances("pytorch")

    onnx_model_path = onnx_runtime_session.pytorch_to_onnx(pytorch_model)

    onnx_model_graph = onnx_runtime_session.load_onnx(onnx_model_path).graph

    onnx_runtime_session.summarize_ONNX_graph(onnx_model_graph)

    if do_test == 'after' or do_test == 'both':
        onnx_runtime_session.test_performances("onnx")

    elif do_test == 'NA':
        pass

    else:
        raise Exception(f"Test argument not recognized; expected one in ['before','after','both','NA'], but {do_test} was received.")
    
    return onnx_model_graph, {}
class ONNXRuntime:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def prepare_save_path(self):
        """Creates and returns a save path for the model."""
        root = Path(__file__).resolve().parents[7]
        current_date = datetime.now().strftime("%Y_%m_%d")
        save_dir = root / f"mase_output/ONNXRuntime" / current_date
        save_dir.mkdir(parents=True, exist_ok=True)

        existing_versions = len(os.listdir(save_dir))
        version = "version_0" if existing_versions==0 else f"version_{existing_versions}"

        save_dir = save_dir / version
        save_dir.mkdir(parents=True, exist_ok=True)

        return save_dir / f"model.onnx"
    
    def pytorch_to_onnx(self, model):
        """Converts PyTorch model to ONNX format and saves it."""
        self.logger.info("Converting PyTorch model to ONNX...")

        save_path = self.prepare_save_path(method='ONNX')
        
        dataloader = self.config['data_module'].train_dataloader()  
        train_sample = next(iter(dataloader))[0]
        train_sample = train_sample.to(self.config['accelerator'])

        torch.onnx.export(model, train_sample, save_path, export_params=True, opset_version=11, 
                            do_constant_folding=True, input_names=['input'])
        self.logger.info(f"ONNX Conversion Complete. Stored ONNX model to {save_path}")

        self.onnx_path = save_path
        
        return save_path

    def summarize_ONNX_graph(self, graph):
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

    def load_onnx(self, onnx_model_path):
        '''Load .onnx model'''
        
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        return onnx_model

    def test_performances(self, model_type):
        if model_type == 'pytorch':
            ...
        elif model_type == 'onnx':
            ...
        else:
            raise Exception(f"Expected model_type being either 'pytorch' or 'onnx', but '{model_type}' received.")

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
    