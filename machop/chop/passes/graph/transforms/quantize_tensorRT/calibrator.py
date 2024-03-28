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
            print(f"{name:40}: {module}")
    model.cuda()


def collect_stats(model, data_loader, num_batches):
    """
    Feed data to the network and collect statistic
    """

    # turn on calibration tool
    for name, module in model.named_modules():
        if isinstance(module, qnn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (xTrain, yTrain) in enumerate(data_loader):
        if i >= num_batches:
            break
        model(Variable(xTrain).cuda())

    # turn off calibration tool
    for name, module in model.named_modules():
        if isinstance(module, qnn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def graph_calibration_pass(graph, pass_args=None):
    """
    Calibrate the model using pytorch quantization calibrator
    """

    # quant_modules.initialize()
    graph.model.cuda()

    collect_stats(
        graph.model,
        pass_args["data_module"].train_dataloader(),
        pass_args["num_batches"],
    )

    if pass_args["calibrator"] == "percentile":
        for percentile in pass_args["percentiles"]:
            compute_amax(graph.model, method="percentile", percentile=percentile)
    else:
        compute_amax(graph.model, method=pass_args["calibrator"])

    print("Succeeded calibrating model in pyTorch!")
    return graph
