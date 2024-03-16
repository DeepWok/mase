import os
from datetime import datetime as dt
from glob import glob
from copy import copy, deepcopy
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import sys
import os
from pathlib import Path
from ......plt_wrapper.base import WrapperBase
from pytorch_lightning import Trainer
from .utils import prepare_save_path


def tensorrt_fine_tune_transform_pass(graph, pass_args=None):
    """Performs Quantized Aware Training"""
    trainer = FineTuning(graph, pass_args)
    ckpt_save_path = trainer.train()

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)

    return graph, {}


class FineTuning:
    def __init__(self, graph, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.graph = graph
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph.model.to(self.device)

    def train(self):
        """
        For QAT it is typical to employ 10% of the original training epochs, 
        starting at 1% of the initial training learning rate, and a cosine 
        annealing learning rate schedule that follows the decreasing half of 
        a cosine period, down to 1% of the initial fine tuning learning rate 
        (0.01% of the initial training learning rate). However this default 
        can be overidden by setting the `epochs`, `initial_learning_rate` and 
        `final_learning_rate` in `passes.tensorrt_quantize.fine_tune`.
        """

        from chop.actions import train
        from chop.models import get_model_info

        # load the settings and default to chop default parameters
        model_info = get_model_info(self.config["data_module"].model_name)
        weight_decay = (
            self.config["weight_decay"] if "weight_decay" in self.config else 0
        )
        optimizer = self.config["optimizer"] if "optimizer" in self.config else "adam"

        # Check if user would like to override the initial learning rate otherwise default to 1% of original LR 
        try:
            initial_fine_tune_lr = (self.config["initial_learning_rate"]) * 0.01
        except KeyError:
            initial_fine_tune_lr = (self.config.get("learning_rate", 1e-5)) * 0.01
        
        # Check if user would like to override the final learning rate otherwise default to 
        # 1% of initial learning rate or 0.01% of original learning rate
        try:
            eta_min = self.config["final_learning_rate"]
        except KeyError:
            eta_min = (initial_fine_tune_lr * 0.01)  # Decreases to 

        # Check if user would like to override the number of epochs otherwise default to 10% of original epochs
        try:
            t_max = self.config["fine_tune"]["epochs"]
        except KeyError:
            t_max = (len(self.config["data_module"].train_dataloader()) * self.config("max_epochs", 20) * 0.1)

        ckpt_save_path = prepare_save_path("ckpts")

        scheduler_args = {
            "t_max": t_max, 
            "eta_min": eta_min
        }

        plt_trainer_args = {
            "max_epochs": self.config["fine_tune"]["epochs"],
            "accelerator": self.config["accelerator"],
        }

        train(
            self.graph.model,
            model_info,
            self.config["data_module"],
            self.config["data_module"].dataset_info,
            "Quantization Fine Tuning",
            optimizer,
            initial_fine_tune_lr,
            weight_decay,
            scheduler_args,
            plt_trainer_args,
            False,
            ckpt_save_path,
            None,
            None,
            "",
        )

        return ckpt_save_path / "best.ckpt"
