import logging
from pathlib import PosixPath

from onnxruntime.quantization import quantize_dynamic, QuantType

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