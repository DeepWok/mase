import os
from datetime import datetime as dt
from glob import glob
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
from .utils import FakeQuantizer

def tensorrt_fake_quantize_transform_pass(graph, pass_args=None):
    by = pass_args["by"]
    fq = FakeQuantizer(pass_args)
    match by:
        case "type":
            graph = fq.fake_quantize_by_type(graph)
        case "name":
            graph = fq.fake_quantize_by_name(graph)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    return graph, {}

def tensorrt_calibrate_transform_pass(graph, pass_args=None):
    calibrator = Calibrator(pass_args)
    graph = calibrator.calibrate_model(graph)
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}

class Calibrator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_config(self, config: dict, name: str):
        """Retrieve specific configuration from the config dictionary or return default."""
        return config.get(name, config['default'])['config']

    def compute_amax(self, model, **kwargs):
        """Computes and loads the maximum activation values for quantization calibration."""
        # Load calibration result
        for name, module in model.named_modules():
            if isinstance(module, qnn.TensorQuantizer):
                if module._calibrator is not None:
                    # Load calibration max values depending on the calibrator type
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
                self.logger.info(f"{name:40}: {module}")
        model.cuda()

    def calibrate_model(self, graph):
        """Performs the calibration pass on the model using the given data loader."""
        self.logger.info("Starting calibration of the model in PyTorch...")
        quant_modules.initialize()
        graph.model.cuda()
        
        with t.no_grad():
            # Turn on calibration tool
            for name, module in graph.model.named_modules():
                if isinstance(module, qnn.TensorQuantizer):
                    if module._calibrator is not None:
                        self.logger.info("Disabling Quantization and Enabling Calibration")
                        module.disable_quant()
                        module.enable_calib()
                    else:
                        module.disable()

            batches = self.config.get('num_calibration_batches', 10)
            dataloader = self.config['data_module'].train_dataloader()

            for i, (xTrain, _) in enumerate(dataloader):
                graph.model(Variable(xTrain).cuda())
                if i >= batches:
                    break           

            # Turn off calibration tool
            for _, module in graph.model.named_modules():
                if isinstance(module, qnn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.enable_quant()
                        self.logger.info("Enabling Quantization and Disabling Calibration")
                        module.disable_calib()
                    else:
                        module.enable()
            
            # Apply the specific calibration based on user input
            try:
                calibs = self.config.get('default')['config']['calibrators']
            except KeyError:
                calibs = 'entropy'
            for calib in calibs:
                match calib:
                    case "entropy":
                        self.compute_amax(graph.model, method=calib)
                    case "percentile":
                        try:
                            percentiles = self.config.get('default')['config']['percentiles']
                        except KeyError:
                            percentiles = [99]
                        for percentile in percentiles:
                            self.compute_amax(graph.model, method=calib)
                    case "mse":
                        self.compute_amax(graph.model, method=calib)

                # perform an analysis pass if required
                if self.config['post_calibration_analysis']:
                    from chop.passes.graph import tensorrt_analysis_pass
                    self.logger.info(f"Performing post calibration analysis for calibrator {calib}...")
                    tensorrt_analysis_pass(graph, pass_args=self.config)
                    self.logger.info("Post calibration analysis complete.")
            
            self.logger.info("Succeeded in calibrating the model in PyTorch!")
            return graph