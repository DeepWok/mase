import os
from datetime import datetime as dt
from glob import glob
from copy import copy, deepcopy
import logging
import numpy as np
import pytorch_quantization.calib as calib
import pytorch_quantization.nn as qnn
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from cuda import cudart
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch.autograd import Variable            
import torch
from typing import Dict
from chop.tools.utils import copy_weights, init_LinearLUT_weight, init_Conv2dLUT_weight
from torch import nn
from pathlib import Path
from datetime import datetime
import time
import pynvml
import threading

from ....utils import (
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

QUANTIZEABLE_OP = (
    # "add",
    # "bmm",
    # "conv1d",
    "conv2d",
    # "matmul",
    # "mul",
    "linear",
    # "relu",
    # "sub",
)
class FakeQuantizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_quantized_module(self,
        mase_op: str,
        original_module: nn.Module,
        config: dict
    ):
        original_module_cls = type(original_module)
        # convert quantize_axis from false to None (since TOML does not support None)
        config["input"]["quantize_axis"] = None if config["input"]["quantize_axis"] == False else config["input"]["quantize_axis"]
        config["weight"]["quantize_axis"] = None if config["weight"]["quantize_axis"] == False else config["weight"]["quantize_axis"]
        #TODO implement more module support: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#quantized-modules
        try:
            # Set default quantization descriptor for input and weights
            qnn.QuantLinear.set_default_quant_desc_input(QuantDescriptor(calib_method=config['config']['calibrator'], axis=config["input"]["quantize_axis"]))
            qnn.QuantLinear.set_default_quant_desc_weight(QuantDescriptor(calib_method=config['config']['calibrator'], axis=config["input"]["quantize_axis"]))
            if mase_op == "linear":
                use_bias = original_module.bias is not None

                new_module = qnn.QuantLinear(
                    in_features=original_module.in_features,
                    out_features=original_module.out_features,
                    bias=use_bias
                )

                copy_weights(original_module.weight, new_module.weight)
                if use_bias:
                    copy_weights(original_module.bias, new_module.bias)
        
            elif mase_op in ("conv2d"):
                # Set default quantization descriptor for input and weights
                qnn.QuantConv2d.set_default_quant_desc_input(QuantDescriptor(calib_method=config['config']['calibrator'], axis=config["input"]["quantize_axis"]))
                qnn.QuantConv2d.set_default_quant_desc_weight(QuantDescriptor(calib_method=config['config']['calibrator'], axis=config["input"]["quantize_axis"]))
                use_bias = original_module.bias is not None
                new_module = qnn.QuantConv2d(
                    in_channels=original_module.in_channels,
                    out_channels=original_module.out_channels,
                    kernel_size=original_module.kernel_size,
                    stride=original_module.stride,
                    padding=original_module.padding,
                    dilation=original_module.dilation,
                    groups=original_module.groups,
                    bias=use_bias,
                    padding_mode=original_module.padding_mode,
                )

                copy_weights(original_module.weight, new_module.weight)
                if use_bias:    
                    copy_weights(original_module.bias, new_module.bias)

            else:
                raise NotImplementedError(
                    f"Unsupported module class {original_module_cls} to modify"
                )
            
        except KeyError:
            raise Exception(f"Config/TOML not configured correctly for layer {original_module_cls}. Please check documentation for what must be defined.")
        
        return new_module
    
    def get_config(self, name: str):
        """Retrieve specific configuration from the instance's config dictionary or return default."""
        config = self.config.get(name)
        if type(config) is type(None):
            raise Exception(f"Please check Config/TOML file. Layer config {name} must be defined.")
        return config

    def fake_quantize_by_type(self, graph):
        """
        This method applies fake quantization to the graph based on the type of each node.
        """
        self.logger.info("Applying fake quantization to PyTorch model...")
        for node in graph.fx_graph.nodes:
            if get_mase_op(node) not in QUANTIZEABLE_OP:
                continue
            node_config = self.get_config(get_mase_op(node))
            if not node_config['config']['quantize']:
                continue
            if node.op == "call_module":
                original_module = get_node_actual_target(node)
                new_module = self.create_quantized_module(
                    get_mase_op(node),
                    original_module,
                    node_config,
                )
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
        self.logger.info("Fake quantization applied to PyTorch model.")
        return graph

    def fake_quantize_by_name(self, graph):
        """
        This method applies fake quantization to the graph based on the name of each node.
        """
        #TODO implement fake quantize_by_name
        return graph

class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, nCalibration, input_generator, cache_file_path):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.nCalibration = nCalibration
        self.shape = next(iter(input_generator))[0].shape
        self.buffer_size = trt.volume(self.shape) * trt.float32.itemsize
        self.cache_file = cache_file_path
        _, self.dIn = cudart.cudaMalloc(self.buffer_size)
        self.input_generator = input_generator
    
    def get_batch_size(self):
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):
        try:
            data = np.array(next(iter(self.input_generator))[0])
            cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffer_size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print("Succeed finding cache file: %s" % (self.cache_file))
            with open(self.cache_file, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")
        return

class PowerMonitor(threading.Thread):
    def __init__(self, config):
        super().__init__()  # Call the initializer of the base class, threading.Thread
        # Initialize the NVIDIA Management Library (NVML)
        pynvml.nvmlInit()
        self.power_readings = []  # List to store power readings
        self.running = False      # Flag to control the monitoring loop
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assume using GPU 0

    def run(self):
        self.running = True
        while self.running:
            # Get current GPU power usage in milliwatts and convert to watts
            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_W = power_mW / 1000.0
            self.power_readings.append(power_W)
            time.sleep(0.001)  # Wait before next reading

    def stop(self):
        self.running = False  # Stop the monitoring loop

def prepare_save_path(method: str):
    """Creates and returns a save path for the model."""
    root = Path(__file__).resolve().parents[7]
    current_date = datetime.now().strftime("%Y_%m_%d")
    save_dir = root / f"mase_output/TensorRT/Quantization/{method}" / current_date
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_versions = len(os.listdir(save_dir))
    version = "version_0" if existing_versions==0 else f"version_{existing_versions}"

    save_dir = save_dir / version
    save_dir.mkdir(parents=True, exist_ok=True)

    if method == 'CKPT':
        return save_dir / "fine_tuning_ckpts"

    return save_dir / f"model.{method.lower()}"

