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

logger = logging.getLogger(__name__)


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, qnn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()


def graph_calibration_pass(graph, dataloader, pass_args=None):
    quant_modules.initialize()
    graph.model.cuda()
    
    with t.no_grad():
        # turn on calibration tool
        for name, module in graph.model.named_modules():
            if isinstance(module, qnn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        for i, (xTrain, yTrain) in enumerate(dataloader):
            if i >= 30:
                break
            graph.model(Variable(xTrain).cuda())

        # turn off calibration tool
        for name, module in graph.model.named_modules():
            if isinstance(module, qnn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

        
        if pass_args['calibrator'] == "percentile":
            for percentile in pass_args['percentiles']:
                compute_amax(graph.model, method="percentile")
        else:
            compute_amax(graph.model, method=pass_args['calibrator'])    

    print("Succeeded calibrating model in pyTorch!")