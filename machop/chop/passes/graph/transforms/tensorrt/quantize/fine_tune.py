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
from chop.actions import train
from chop.models import get_model_info, get_model

def tensorrt_fine_tune_transform_pass(graph, pass_args=None):
    """ Performs Quantized Aware Training """
    trainer = FineTuning(graph, pass_args)
    graph = trainer.train()
    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}

class FineTuning:
    def __init__(self, graph, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize the optimizer.
        self.learning_rate = self.config['learning_rate'] if 'learning_rate' in self.config else 1e-5
        self.weight_decay = self.config['weight_decay'] if 'weight_decay' in self.config else 0
        plt_trainer_args = {
            "max_epochs": self.config['fine_tune']['qat_epochs'],
            "accelerator": self.config['accelerator'],
            }
        initial_learning_rate = self.learning_rate * 0.01  # Start with 1% of the original LR
        optimizer = torch.optim.Adam(self.graph.parameters(), lr=initial_learning_rate)
        
        # Set up the scheduler
        self.num_steps = len(self.config['data_module'].train_dataloader())
        self.total_steps =  self.num_steps * self.config['qat_epochs']
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.total_steps, eta_min=initial_learning_rate * 0.01)  # Decreases to 0.01% of initial

    def train(self):        
        train(self.graph.model, get_model_info(self.config['model']), self.config['data_module'], self.config['data_module'].dataset_info, 'Quantization Fine Tuning', self.optimizer, self.learning_rate, self.weight_decay, self.plt_trainer_args, False, './ckpts/quant-fine-tuning', None, None, "")