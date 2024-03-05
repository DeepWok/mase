from copy import copy, deepcopy
import logging
import torch
import os
import sys
from pathlib import Path
from datetime import datetime
import tensorrt as trt
from pathlib import Path
from datetime import datetime

from pytorch_quantization import quant_modules, calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor

from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
from ....utils import deepcopy_mase_graph


def tensorrt_transformation_analysis_pass(original_graph, quantized_graph, pass_args=None):
    analysis = Analaysis(original_graph, quantized_graph, pass_args)
    analysis.evaluate()


class Analaysis:
    def __init__(self, original_graph, quantized_graph, config):
        self.original_graph = original_graph
        self.quantized_graph = quantized_graph
        self.config = config
        self.logger = logging.getLogger(__name__)

    def evaluate(self):
        self.logger.info("Starting TensorRT transformation analysis")
        for analysis in self.config["analysis"]:
            match analysis:
                case "latency":
                    self._eval_latency()
                case "accuracy":
                    self._eval_accuracy()
                case _:
                    raise ValueError(f"Unsupported analysis: {analysis}")
        self.logger.info("Finished TensorRT transformation analysis")

    def _eval_latency(self):
        self.logger.info("Evaluating latency")
        

    def _eval_accuracy(self):
        self.logger.info("Evaluating accuracy")
