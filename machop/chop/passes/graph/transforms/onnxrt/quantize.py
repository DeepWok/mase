from onnxruntime.quantization import quantize_dynamic, QuantType

QUANT_MAP = {
    "int8": QuantType.QInt8,
    "uint8": QuantType.QUInt8,
    "int16": QuantType.QInt16,
    "uint16": QuantType.QUInt16,
}

class Quantizer:
    def __init__(self, config):
        self.config = config

    def quantize_dynamic(self, model):
        """Quantize the model using dynamic quantization."""
        self.logger.info("Quantizing model using dynamic quantization...")

        try:
            precision = QUANT_MAP(self.config['default']['config']['precision'])
        except KeyError:
            precision = QUANT_MAP("int8")

        quantized_model = quantize_dynamic(
            self.model,
            self.quantization_params,
            weight_type=precision, 
        )

        return quantized_model