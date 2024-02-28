from copy import copy, deepcopy
import logging
import torch
import os
import sys
from pathlib import Path
from datetime import datetime
import tensorrt as trt

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# pip install --no-cache-dir --index-url https://pypi.nvidia.com pytorch-quantization !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#  ./ch transform --config configs/examples/toy_uniform_tensorRT.toml --load /home/wfp23/ADL/Mauro/mase/mase_output/jsc-tiny_classification_jsc_2024-02-20/software/training_ckpts/best.ckpt --load-type pl
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from pytorch_quantization import quant_modules, calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor


from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
from ....utils import deepcopy_mase_graph


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]
    
def pre_quantization_test(model):
    print("Evaluate pre-quantization performance...")
    ...

def pytorch_quantize_by_type(graph, config):
    ...

def pytorch_to_onnx(model, config):
    print("Converting PyTorch model to ONNX...")

    # Prepare the save path
    root = Path(__file__).resolve().parents[7]
    current_date = datetime.now().strftime("%Y_%m_%d")
    save_dir = root / "mase_output/TensorRT/Quantization/ONNX" / current_date
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract existing version numbers
    existing_versions = [int(d.name.split("_")[-1]) for d in save_dir.parent.iterdir() if d.is_dir() and d.name.startswith(current_date) and d.name.startswith("version_")]
    version = "version_0" if not existing_versions else f"version_{max(existing_versions) + 1}"

    save_dir = save_dir / version
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "model.onnx"
    import pdb; pdb.set_trace()

    dataloader = config['train_generator'].dataloader    
    train_sample = next(iter(dataloader))[0]
    train_sample.to(config['device'])

    torch.onnx.export(model, train_sample, save_path,
                    export_params=True, opset_version=11, do_constant_folding=True,
                    input_names = ['input'])
    print(f"ONNX model saved to {save_path}")
    print("Conversion to ONNX done!")
    return save_path


def calibrate_onnx(model, train_loader, onnx_path):
    print("Calibrating model for quantization...") 
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20  # Adjust size as needed
    engine = builder.build_engine(network, config)

    with open(onnx_path / 'trt', 'wb') as f:
        f.write(engine.serialize())   

    print("Calibration done!")

def save_tensorrt_model(model, output_dir):
    print(f"Saving quantized TensorRT model in {output_dir}")
    ...
    
def quantize():
    ...

def convert_to_quantized_node(node):
    match get_node_actual_target(node):
        case 'torch.nn.modules.batchnorm.BatchNorm1d':
            return 


def pytorch_quantize_by_type(graph, config: dict):  
    if 'default' in config:
        model = graph.model.eval()
        import pdb
        for node in graph.nodes:
            convert_to_quantized_node(node)
            pdb.set_trace()

        # model = graph.model.eval()
        # onnx_path = pytorch_to_onnx(model, config)
        # calibrate_onnx(model, config['train_generator'].dataloader, onnx_path)  
        

    return graph

    



def pytorch_quantize_by_name(graph, config: dict):
    raise NotImplementedError()


def pytorch_quantize_by_regex_name(graph, config: dict):
    raise NotImplementedError()


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
