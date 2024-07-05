import os
import logging
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Dict
from chop.tools.utils import copy_weights
from torch import nn
from pathlib import Path
from datetime import datetime
import time
import threading

from ....utils import (
    get_mase_op,
    get_node_actual_target,
    get_parent_name,
)


from cuda import cudart
from pytorch_quantization.tensor_quant import QuantDescriptor
import pynvml

import pytorch_quantization.nn as qnn
import tensorrt as trt

QUANTIZEABLE_OP = {
    "conv1d": qnn.QuantConv1d,
    "conv2d": qnn.QuantConv2d,
    "conv3d": qnn.QuantConv3d,
    "convTranspose1d": qnn.QuantConvTranspose1d,
    "convTranspose2d": qnn.QuantConvTranspose2d,
    "convTranspose3d": qnn.QuantConvTranspose3d,
    "linear": qnn.QuantLinear,
    "avgPool1d": qnn.QuantAvgPool1d,
    "avgPool2d": qnn.QuantAvgPool2d,
    "avgPool3d": qnn.QuantAvgPool3d,
    "maxPool1d": qnn.QuantMaxPool1d,
    "maxPool2d": qnn.QuantMaxPool2d,
    "maxPool3d": qnn.QuantMaxPool3d,
}


class FakeQuantizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_quantized_module(
        self, mase_op: str, original_module: nn.Module, config: dict
    ):
        original_module_cls = type(original_module)
        # convert quantize_axis from false to None (since TOML does not support None)
        config["input"]["quantize_axis"] = (
            None
            if config["input"]["quantize_axis"] == False
            else config["input"]["quantize_axis"]
        )
        config["weight"]["quantize_axis"] = (
            None
            if config["weight"]["quantize_axis"] == False
            else config["weight"]["quantize_axis"]
        )
        try:
            op = QUANTIZEABLE_OP[mase_op]
        except:
            raise Exception(
                f"Module {original_module_cls} unsupported. Please check documentation for what is currently supported."
            )
        try:
            match mase_op:
                case "linear":
                    use_bias = original_module.bias is not None

                    new_module = op(
                        in_features=original_module.in_features,
                        out_features=original_module.out_features,
                        bias=use_bias,
                    )
                    # Set default quantization descriptor for input and weights
                    op.set_default_quant_desc_input(
                        QuantDescriptor(
                            calib_method=config["input"]["calibrator"],
                            axis=config["input"]["quantize_axis"],
                        )
                    )
                    op.set_default_quant_desc_weight(
                        QuantDescriptor(
                            calib_method=config["weight"]["calibrator"],
                            axis=config["input"]["quantize_axis"],
                        )
                    )

                    copy_weights(original_module.weight, new_module.weight)
                    if use_bias:
                        copy_weights(original_module.bias, new_module.bias)

                case (
                    "conv1d"
                    | "conv2d"
                    | "conv3d"
                    | "convTranspose1d"
                    | "convTranspose2d"
                    | "convTranspose3d"
                ):
                    # Set default quantization descriptor for input and weights
                    op.set_default_quant_desc_input(
                        QuantDescriptor(
                            calib_method=config["input"]["calibrator"],
                            axis=config["input"]["quantize_axis"],
                        )
                    )
                    op.set_default_quant_desc_weight(
                        QuantDescriptor(
                            calib_method=config["weight"]["calibrator"],
                            axis=config["input"]["quantize_axis"],
                        )
                    )
                    use_bias = original_module.bias is not None
                    new_module = op(
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

                case (
                    "avgPool1d"
                    | "avgPool2d"
                    | "avgPool3d"
                    | "maxPool1d"
                    | "maxPool2d"
                    | "maxPool3d"
                ):
                    # Set default quantization descriptor for input since pooling layers typically do not have weights
                    op.set_default_quant_desc_input(
                        QuantDescriptor(
                            calib_method=config["input"]["calibrator"],
                            axis=config["input"]["quantize_axis"],
                        )
                    )

                    # Configure new pooling module with parameters from the original module
                    new_module = op(
                        kernel_size=original_module.kernel_size,
                        stride=original_module.stride,
                        padding=original_module.padding,
                        dilation=(
                            original_module.dilation
                            if hasattr(original_module, "dilation")
                            else None
                        ),  # Not all pooling layers have a dilation attribute
                        return_indices=(
                            original_module.return_indices
                            if hasattr(original_module, "return_indices")
                            else False
                        ),  # Only relevant for max pooling
                        ceil_mode=original_module.ceil_mode,
                    )

                case "LSTM":
                    new_module = QUANTIZEABLE_OP["LSTM"](
                        input_size=original_module.input_size,
                        hidden_size=original_module.hidden_size,
                        num_layers=original_module.num_layers,
                        bias=original_module.bias,
                        batch_first=original_module.batch_first,
                        dropout=original_module.dropout,
                        bidirectional=original_module.bidirectional,
                    )

                    # Check the number of layers and bidirectional configuration
                    num_layers = original_module.num_layers
                    bidirectional = 2 if original_module.bidirectional else 1

                    for layer in range(num_layers):
                        for direction in range(bidirectional):
                            # Suffix to identify the parameters for the layer and direction
                            suffix = f"_reverse" if direction == 1 else ""
                            layer_idx = f"{layer}{suffix}"

                            # Copy weights for the input-hidden connections (ih)
                            attr = f"weight_ih_l{layer_idx}"
                            getattr(new_module, attr).data.copy_(
                                getattr(original_module, attr).data
                            )

                            # Copy weights for the hidden-hidden connections (hh)
                            attr = f"weight_hh_l{layer_idx}"
                            getattr(new_module, attr).data.copy_(
                                getattr(original_module, attr).data
                            )

                            # Copy biases, if they exist
                            if original_module.bias:
                                attr = f"bias_ih_l{layer_idx}"
                                getattr(new_module, attr).data.copy_(
                                    getattr(original_module, attr).data
                                )

                                attr = f"bias_hh_l{layer_idx}"
                                getattr(new_module, attr).data.copy_(
                                    getattr(original_module, attr).data
                                )

                case "LSTMCell":
                    new_module = QUANTIZEABLE_OP["LSTMCell"](
                        input_size=original_module.input_size,
                        hidden_size=original_module.hidden_size,
                        bias=original_module.bias,
                    )

                    # Copy weights and biases from the original module to the new quantized module
                    copy_weights(original_module.weight_ih, new_module.weight_ih)
                    copy_weights(original_module.weight_hh, new_module.weight_hh)
                    if original_module.bias:
                        copy_weights(original_module.bias_ih, new_module.bias_ih)
                        copy_weights(original_module.bias_hh, new_module.bias_hh)

                case _:
                    raise NotImplementedError(
                        f"Unsupported module class {original_module_cls} to modify"
                    )

        except KeyError:
            raise Exception(
                f"Config/TOML not configured correctly for layer {original_module_cls}. Please check documentation for what must be defined."
            )

        return new_module

    def get_config(self, name: str):
        """Retrieve specific configuration from the instance's config dictionary or return default."""
        try:
            config = self.config.get(name) or self.config.get("default")
            config["input"] = config.get("input", self.config["default"].get("input"))
            config["weight"] = config.get(
                "weight", self.config["default"].get("weight")
            )

        except KeyError:
            raise Exception(
                f"Please check Config/TOML file. Default or layer {name} config must be defined."
            )

        # Check if required keys are defined
        try:
            config["config"]["quantize"] = config["config"].get(
                "quantize", self.config.get("default")["config"]["quantize"]
            )
            config["config"]["precision"] = config["config"].get(
                "precision", self.config.get("default")["config"]["precision"]
            )
            config["input"]["calibrator"] = config["input"].get(
                "calibrator", self.config.get("default")["input"]["calibrator"]
            )
            config["weight"]["calibrator"] = config["weight"].get(
                "calibrator", self.config.get("default")["weight"]["calibrator"]
            )
        except KeyError:
            raise Exception(
                f"Config/TOML not configured correctly. Please check documentation for what must be defined."
            )
        return config

    def fake_quantize_by_type(self, graph):
        """
        This method applies fake quantization to the graph based on the type of each node.
        """
        self.logger.info("Applying fake quantization to PyTorch model...")

        if not check_for_value_in_dict(self.config, "int8"):
            self.logger.warning(
                "int8 precision not found in config. Skipping fake quantization."
            )
            return graph

        for node in graph.fx_graph.nodes:
            if get_mase_op(node) not in QUANTIZEABLE_OP:
                continue
            node_config = self.get_config(get_mase_op(node))
            if not node_config["config"]["quantize"]:
                continue
            if not node_config["config"]["precision"] == "int8":
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
        This method applies fake quantization to the graph based on the type of each node.
        """
        self.logger.info("Applying fake quantization to PyTorch model...")

        if not check_for_value_in_dict(self.config, "int8"):
            self.logger.warning(
                "int8 precision not found in config. Skipping fake quantization."
            )
            return graph

        for node in graph.fx_graph.nodes:
            if get_mase_op(node) not in QUANTIZEABLE_OP:
                continue
            node_config = self.get_config(node.name)
            if not node_config["config"]["quantize"]:
                continue
            if not node_config["config"]["precision"] == "int8":
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


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
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
            cudart.cudaMemcpy(
                self.dIn,
                data.ctypes.data,
                self.buffer_size,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
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
        self.running = False  # Flag to control the monitoring loop
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assume using GPU 0

    def run(self):
        self.running = True
        while self.running:
            # Get current GPU power usage in milliwatts and convert to watts
            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_W = power_mW / 1000.0
            self.power_readings.append(power_W)
            time.sleep(0.001)  # Wait before next readi

    def stop(self):
        self.running = False  # Stop the monitoring loop


def prepare_save_path(config, method: str, suffix: str):
    """Creates and returns a save path for the model."""
    root = Path(__file__).resolve().parents[7]
    current_date = datetime.now().strftime("%Y-%m-%d")
    model_dir = f'{config["model"]}_{config["task"]}_{config["dataset"]}_{current_date}'
    save_dir = root / f"mase_output/tensorrt/quantization/{model_dir}" / current_date
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_versions = len(os.listdir(save_dir))
    version = "version_0" if existing_versions == 0 else f"version_{existing_versions}"

    save_dir = save_dir / version
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir / f"model.{suffix}"


def check_for_value_in_dict(d, value):
    """Checks if a value is in a hierarchical dictionary."""
    if isinstance(d, dict):  # Check if it's a dictionary
        for k, v in d.items():
            if v == value:  # Check if the value matches
                return True
            elif isinstance(
                v, (dict, list)
            ):  # If the value is a dict or list, search recursively
                if check_for_value_in_dict(v, value):
                    return True
    elif isinstance(d, list):  # Check if it's a list
        for item in d:
            if check_for_value_in_dict(item, value):  # Recurse for each item
                return True
    return False


def get_calibrator_dataloader(original_dataloader, num_batches=200):
    # Get the batch size from the original DataLoader
    batch_size = original_dataloader.batch_size

    # Calculate the number of samples needed for the desired number of batches
    num_samples = num_batches * batch_size

    # Assuming the dataset is accessible through the DataLoader
    original_dataset = original_dataloader.dataset

    # Create a subset of the original dataset
    # Note: This assumes that indexing the dataset returns individual samples.
    # If your dataset returns batches, this approach needs to be adjusted.
    subset_dataset = Subset(original_dataset, range(num_samples))

    # Create a new DataLoader from the subset dataset
    calibrator_dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,  # Typically calibration data isn't shuffled, adjust as needed.
        num_workers=original_dataloader.num_workers,
        pin_memory=original_dataloader.pin_memory,
    )

    return calibrator_dataloader
