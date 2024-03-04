import os
from datetime import datetime as dt
from glob import glob
import logging

import cv2
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


def tensorrt_calibrate_transform_pass(graph, pass_args=None):
    by = pass_args.pop("by")
    calibrator = Calibrator(pass_args)
    match by:
        case "type":
            graph = calibrator.calibrate_model_by_type(graph)
        case "name":
            ...
        case "regex_name":
            ...
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}


class Calibrator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_config(self, config: dict, name: str):
        """Retrieve specific configuration from the config dictionary or return default."""
        return config.get(name, config['default'])['config']

    def compute_activation_max(self, model, **kwargs):
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

    def calibrate_model_by_type(self, graph):
        """Performs the calibration pass on the model using the given data loader."""
        dataloader = self.config['data_loader']
        quant_modules.initialize()
        graph.model.cuda()
        
        with t.no_grad():
            # Turn on calibration tool
            for name, module in graph.model.named_modules():
                if isinstance(module, qnn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.disable_quant()
                        module.enable_calib()
                    else:
                        module.disable()

            for i, (xTrain, yTrain) in enumerate(dataloader):
                if i >= 30:  # Limit the number of batches to process
                    break
                graph.model(Variable(xTrain).cuda())

            # Turn off calibration tool
            for name, module in graph.model.named_modules():
                if isinstance(module, qnn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.enable_quant()
                        module.disable_calib()
                    else:
                        module.enable()
            
            # Apply the specific calibration based on user input
            if self.config:
                match self.config['calibrator']:
                    case "percentile":
                        for percentile in self.config.get('percentiles', [99]):
                            self.compute_activation_max(graph.model, method="percentile")
                    case _:
                        self.compute_activation_max(graph.model, method=self.config.get('calibrator', 'max'))

            self.logger.info("Succeeded in calibrating the model in PyTorch!")
            return graph