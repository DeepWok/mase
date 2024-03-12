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

def tensorrt_train_transform_pass(graph, pass_args=None):
    """ Performs Quantized Aware Training """
    trainer = FineTuning(graph, pass_args)
    graph = trainer.train()
    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}

class FineTuning:
    def __init__(self, graph, config):
        self.logger = config['logger']
        self.config = config
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize the optimizer.
        initial_learning_rate = self.config['learning_rate'] * 0.01  # Start with 1% of the original LR
        #TODO find masegraph optimizer
        self.optimizer = torch.optim.Adam(self.graph.parameters(), lr=initial_learning_rate)
        
        # Set up the scheduler
        self.num_steps = len(self.config['data_module'].train_dataloader())
        self.total_steps =  self.num_steps * self.config['qat_epochs']
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.total_steps, eta_min=initial_learning_rate * 0.01)  # Decreases to 0.01% of initial

    def train(self):
        self.graph.train()  # Set the model to training mode
        for epoch in range(self.epochs):
            for batch_idx, (xs, ys) in enumerate(self.config['data_module'].train_dataloader()):
                xs, ys = xs.to(self.config['accelerator']), ys.to(self.config['accelerator'])
                
                # Forward pass
                outputs = self.graph(xs)
                loss = self.criterion(outputs, ys)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()  # Update the learning rate
                
                if batch_idx % 5 == 0:
                    self.logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}], Step [{batch_idx+1}/{self.num_steps}], Loss: {loss.item():.4f}')
        return self.graph