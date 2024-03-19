import logging
import torch
import json
from pathlib import PosixPath
import tensorrt as trt
from chop.ir import MaseGraph
from .utils import PowerMonitor, prepare_save_path
import sys
import logging 
import os
from tabulate import tabulate
import torch
import torchmetrics
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import onnxruntime as ort
from cuda import cudart
from ....analysis.runtime.analysis import RuntimeAnalysis

def runtime_analysis_pass(model, pass_args=None):
    analysis = RuntimeAnalysis(model, pass_args)
    results = analysis.evaluate()

    results_path = prepare_save_path(method='analysis', suffix='json')

    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    return model, results