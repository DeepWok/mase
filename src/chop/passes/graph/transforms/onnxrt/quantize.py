import logging
import warnings
import os
from pathlib import PosixPath
import torch
import onnx
import onnxruntime
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    shape_inference,
)
from onnxruntime import InferenceSession, SessionOptions
from onnxconverter_common import auto_mixed_precision
from .calibrate import StaticCalibrationDataReader
from torch.utils.data import DataLoader, Subset
from .utils import (
    get_execution_provider,
    get_calibrator_dataloader,
    convert_dataloader_to_onnx_dataset_dict,
)

QUANT_MAP = {
    "int8": QuantType.QInt8,
    "uint8": QuantType.QUInt8,
    # "int16": QuantType.QInt16,
    # "uint16": QuantType.QUInt16,
}


class Quantizer:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config

    def pre_process(self, model_path: PosixPath, prep_path: PosixPath):
        # pre-process the model adding further optimizations
        shape_inference.quant_pre_process(
            str(model_path), str(prep_path), skip_symbolic_shape=False
        )

    def quantize_dynamic(self, model_path: PosixPath, quantized_model_path: PosixPath):
        """Quantize the model using dynamic quantization."""
        self.logger.info("Quantizing model using dynamic quantization...")

        model_path = str(model_path)
        quantized_model_path = str(quantized_model_path)
        try:
            precision = QUANT_MAP[self.config["default"]["config"]["precision"]]
        except KeyError:
            raise Exception(
                "Unsupported or missing precision set in config. Please set a supported precision in the config file."
            )

        quantized_model = quantize_dynamic(
            model_path,
            quantized_model_path,
            weight_type=precision,
        )

        self.logger.info("Quantization complete. Model is now dynamically quantized.")

        return quantized_model

    def quantize_static(self, model_path: PosixPath, quantized_model_path: PosixPath):
        """Quantize the model using dynamic quantization."""
        self.logger.info(
            "Quantizing model using static quantization with calibration..."
        )

        model_path = str(model_path)
        quantized_model_path = str(quantized_model_path)
        try:
            precision = QUANT_MAP[self.config["default"]["config"]["precision"]]
        except KeyError:
            raise Exception(
                "Unsupported or missing precision set in config. Please set a supported precision in the config file."
            )

        if self.config["accelerator"] == "cpu":
            # Get the number of CPU cores
            num_cores = os.cpu_count()

            # Create session options
            sess_options = SessionOptions()

            # Set the number of threads for inter and intra operations based on the CPU core count
            sess_options.inter_op_num_threads = num_cores
            sess_options.intra_op_num_threads = num_cores

            ort_session = InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=[get_execution_provider(self.config["accelerator"])],
            )
        else:
            ort_session = InferenceSession(
                model_path,
                providers=[get_execution_provider(self.config["accelerator"])],
            )

        # Create a calibrator_dataloader that is a subset of the training dataloader
        # Number of batches defined in the config by num_calibration_batches
        calibrator_dataloader = get_calibrator_dataloader(
            self.config["data_module"].train_dataloader(),
            self.config.get("num_calibration_batches", 200),
        )

        data_reader = StaticCalibrationDataReader(
            calibrator_dataloader, input_name=ort_session.get_inputs()[0].name
        )

        quantized_model = quantize_static(
            model_path,
            quantized_model_path,
            data_reader,
            activation_type=precision,
            weight_type=precision,
        )

        self.logger.info(
            "Quantization complete. Model is now calibrated and statically quantized."
        )

        return quantized_model

    def quantize_auto_mixed_precision(
        self, model_path: PosixPath, quantized_model_path: PosixPath
    ):
        """Quantize the model using mixed precision quantization of FP16 and FP32."""
        self.logger.info(
            "Quantizing model using automatic mixed precision quantization..."
        )

        # Load the model
        model_path = str(model_path)
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        # Get the configuration settings if they exist
        config_defaults = self.config.get("default", {}).get("config", {})
        rtol = config_defaults.get("rtol", 0.1)
        atol = config_defaults.get("atol", 0.01)
        keep_io_types = config_defaults.get("keep_io_types", True)

        # convert validation dataloader to a numpy array
        val_data = convert_dataloader_to_onnx_dataset_dict(
            self.config["data_module"].val_dataloader(), ["input"]
        )
        # supress warnings regarding F32 truncation
        warnings.filterwarnings(
            "ignore",
            message="the float32 number .* will be truncated to .*",
            category=UserWarning,
        )
        quantized_model = auto_mixed_precision.auto_convert_mixed_precision(
            model, val_data[0], rtol=rtol, atol=atol, keep_io_types=keep_io_types
        )

        # Save the quantized model
        onnx.save(quantized_model, str(quantized_model_path))

        self.logger.info(
            "Quantization complete. Model is now quantized using automatic mixed precision."
        )

        return quantized_model
