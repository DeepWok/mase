import logging
import os
from pathlib import PosixPath
import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, shape_inference
from onnxruntime import InferenceSession, SessionOptions
from onnxconverter_common import auto_mixed_precision
from .calibrate import StaticCalibrationDataReader
from torch.utils.data import DataLoader, Subset
from .utils import get_execution_provider, get_calibrator_dataloader, convert_dataloader_to_numpy

QUANT_MAP = {
    "int8": QuantType.QInt8,
    "uint8": QuantType.QUInt8,
    "int16": QuantType.QInt16,
    "uint16": QuantType.QUInt16,
}

class Quantizer:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config

    def pre_process(self, model_path:PosixPath, prep_path:PosixPath):
        # pre-process the model adding further optimizations
        shape_inference.quant_pre_process(str(model_path), str(prep_path), skip_symbolic_shape=False)

    def quantize_dynamic(self, model_path:PosixPath, quantized_model_path:PosixPath):
        """Quantize the model using dynamic quantization."""
        self.logger.info("Quantizing model using dynamic quantization...")
        
        model_path = str(model_path)
        quantized_model_path = str(quantized_model_path)
        try:
            precision = QUANT_MAP[self.config['default']['config']['precision']]
        except KeyError:
            raise Exception("Unsupported or missing precision set in config. Please set a supported precision in the config file.")

        quantized_model = quantize_dynamic(
            model_path,
            quantized_model_path,
            weight_type=precision, 
        )

        self.logger.info("Quantization complete. Model is now dynamically quantized.")

        return quantized_model
    
    def quantize_static(self, model_path:PosixPath, quantized_model_path:PosixPath):
        """Quantize the model using dynamic quantization."""
        self.logger.info("Quantizing model using static quantization with calibration...")
        
        model_path = str(model_path)
        quantized_model_path = str(quantized_model_path)
        try:
            precision = QUANT_MAP[self.config['default']['config']['precision']]
        except KeyError:
            raise Exception("Unsupported or missing precision set in config. Please set a supported precision in the config file.")

        if self.config['accelerator'] == 'cpu':
            # Get the number of CPU cores
            num_cores = os.cpu_count()

            # Create session options
            sess_options = SessionOptions()

            # Set the number of threads for inter and intra operations based on the CPU core count
            sess_options.inter_op_num_threads = num_cores
            sess_options.intra_op_num_threads = num_cores

            ort_session = InferenceSession(model_path, sess_options=sess_options, providers=[get_execution_provider(self.config)])
        else:
            ort_session = InferenceSession(model_path, providers=[get_execution_provider(self.config)])

        # Create a calibrator_dataloader that is a subset of the training dataloader
        # Number of batches defined in the config by num_calibration_batches
        calibrator_dataloader = get_calibrator_dataloader(self.config['data_module'].train_dataloader(), self.config.get('num_calibration_batches', 200))

        data_reader = StaticCalibrationDataReader(calibrator_dataloader, input_name=ort_session.get_inputs()[0].name)
        
        quantized_model = quantize_static(
            model_path,
            quantized_model_path,
            data_reader,
            weight_type=precision,
        )

        self.logger.info("Quantization complete. Model is now calibrated and statically quantized.")

        return quantized_model

    def quantize_auto_mixed_precision(self, model_path:PosixPath, quantized_model_path:PosixPath):
        """Quantize the model using mixed precision quantization of FP16 and FP32."""
        self.logger.info("Quantizing model using automatic mixed precision quantization...")

        model_path = str(model_path)
        model = onnx.load(model_path)

        # convert validation dataloader to a numpy array 
        converted_dataloader = convert_dataloader_to_numpy(self.config['data_module'].val_dataloader())
        quantized_model = auto_mixed_precision.auto_convert_mixed_precision(model, converted_dataloader, rtol=0.01, atol=0.001, keep_io_types=True)
        
        quantized_model_path = str(quantized_model_path)
        onnx.save(quantized_model, quantized_model_path)

        self.logger.info("Quantization complete. Model is now quantized using automatic mixed precision.")

        return quantized_model

