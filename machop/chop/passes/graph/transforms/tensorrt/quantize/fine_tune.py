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
    """ Performs Quantized Aware Training """
    trainer = FineTuning(graph, pass_args)
    ckpt_save_path = trainer.train()

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)

    return graph, {}

class FineTuning():
    def __init__(self, graph, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph.model.to(self.device)
            
    def train(self):
        from chop.actions import train  # Rename to avoid confusion with the method name
        from chop.models import get_model_info

        # Optionally, get model information if required by your custom actions or for logging
        model_info = get_model_info(self.config['data_module'].model_name)

        # Initialize the optimizer.
        weight_decay = self.config['weight_decay'] if 'weight_decay' in self.config else 0
        optimizer = self.config['optimizer'] if 'optimizer' in self.config else 'adam'

        initial_fine_tune_lr = (self.config['learning_rate'] if 'learning_rate' in self.config else 1e-5) * 0.01  # Start with 1% of the original LR
        t_max = len(self.config['data_module'].train_dataloader()) * self.config['fine_tune']['epochs'] # Total steps
        eta_min= initial_fine_tune_lr * 0.01 # Decreases to 1% of initial learning rate or 0.01% of original learning rate  

        ckpt_save_path = prepare_save_path('ckpts')

        scheduler_args = {
            "t_max": t_max,
            "eta_min": eta_min
        }

        plt_trainer_args = {
            "max_epochs": self.config['fine_tune']['epochs'],
            "accelerator": self.config['accelerator'],
        }

        train(
            self.graph.model,
            model_info,
            self.config['data_module'],
            self.config['data_module'].dataset_info,
            'Quantization Fine Tuning',
            optimizer,
            initial_fine_tune_lr,
            weight_decay,
            scheduler_args,
            plt_trainer_args,
            False,
            ckpt_save_path,
            None,
            None,
            ""
         )

        return ckpt_save_path / 'best.ckpt'
