import logging
import torch
import os
from datetime import datetime
from pathlib import Path
from prettytable import PrettyTable

import numpy as np

import onnx
import onnxruntime as ort
from .quantize import Quantizer

def onnx_runtime_transform_pass(graph, pass_args="None"):    
    onnx_runtime_session = ONNXRuntime(config=pass_args)

    pytorch_model = graph.model

    onnx_model_path = onnx_runtime_session.pytorch_to_onnx(pytorch_model)
    onnx_model_graph = onnx_runtime_session.load_onnx(onnx_model_path).graph
    onnx_runtime_session.summarize_ONNX_graph(onnx_model_graph)
    quant_meta = onnx_runtime_session.quantize(onnx_model_path)

    return graph, {'onnx_path': onnx_model_path, **quant_meta}


class ONNXRuntime:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _prepare_save_path(self, quantized_type:str):
        """Creates and returns a save path for the model."""
        root = Path(__file__).resolve().parents[6]
        current_date = datetime.now().strftime("%Y_%m_%d")
        save_dir = root / f"mase_output/onnx_runtime/{quantized_type}" / current_date
        save_dir.mkdir(parents=True, exist_ok=True)

        existing_versions = len(os.listdir(save_dir))
        version = (
            "version_0" if existing_versions == 0 else f"version_{existing_versions}"
        )

        save_dir = save_dir / version
        save_dir.mkdir(parents=True, exist_ok=True)

        return save_dir / f"model.onnx"

    def pytorch_to_onnx(self, model):
        """Converts PyTorch model to ONNX format and saves it."""
        self.logger.info("Converting PyTorch model to ONNX...")

        save_path = self._prepare_save_path("unquantized")

        dataloader = self.config["data_module"].train_dataloader()
        train_sample = next(iter(dataloader))[0]
        train_sample = train_sample.to(self.config["accelerator"])

        torch.onnx.export(
            model,
            train_sample,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        self.logger.info(f"ONNX Conversion Complete. Stored ONNX model to {save_path}")

        self.onnx_path = save_path

        return save_path

    def summarize_ONNX_graph(self, graph):
        # Initialize a PrettyTable to display the summary
        summary_table = PrettyTable()
        summary_table.field_names = ["Index", "Name", "Type", "Inputs", "Outputs", "Attributes"]

        # Parse through the model's graph
        for index, node in enumerate(graph.node):
            # Gather node information
            node_name = node.name or f"Node_{index}"  # Some nodes might not have names
            node_type = node.op_type
            inputs = [str(input) for input in node.input]
            outputs = [str(output) for output in node.output]
            attributes = [attr.name for attr in node.attribute]

            # Add information to the table
            summary_table.add_row([index, node_name, node_type, ', '.join(inputs), ', '.join(outputs), ', '.join(attributes)])
        self.logger.info(f"ONNX Model Summary: \n{summary_table}")
    
    def quantize(self, model_path) -> dict:
        if 'quantize_type' in self.config['default']['config']:
            quantizer = Quantizer(self.config)
            try:
                self.config['default']['config']['quantize']
            except:
                self.config['default']['config']['quantize'] = False
                self.logger.warning("quantization is not set in default config. Skipping quantization.")
            try:
                quant_type = self.config['default']['config']['quantize_type']
            except (TypeError, KeyError):
                quant_type = 'static'

            match quant_type:
                case 'static':
                    quantized_path = self._prepare_save_path("static_quantized")
                    quantizer.quantize_static(model_path, quantized_path)
                    return {'onnx_static_quantized_path': quantized_path}
                case 'dynamic':
                    quantized_path = self._prepare_save_path("dynamic_quantized")
                    quantizer.quantize_dynamic(model_path, quantized_path)
                    return {'onnx_dynamic_quantized_path': quantized_path}
                case _:
                    raise Exception(f"Invalid quantization type: {quant_type}")
        return {}

    def load_onnx(self, onnx_model_path):
        """Load .onnx model"""

        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        return onnx_model

    def _get_execution_provider(self):
        EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        return (
            "CUDAExecutionProvider"
            if self.config["accelerator"] == "cuda"
            else "CPUExecutionProvider"
        )
