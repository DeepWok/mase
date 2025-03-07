import copy
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import math
# from clip.model import QuickGELU
# from torch.autograd import Variable
# from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
# import random

def replace_gelu_with_relu(model,convert_layers,num_neurons=64, path=None, device='cpu'):
    for m_str in convert_layers:
        eval(f'model.{m_str}').mlp.gelu = get_distilled_gelu(float16=False,device=device,num_neurons=num_neurons,path=path)



def get_distilled_gelu(device='cuda', float16=True, num_neurons=8, path=None):
	if float16:
		return Distilled_GELU(load_distilled_weights=True,num_neurons=num_neurons,path=path).half().to(device)
	return Distilled_GELU(load_distilled_weights=True,num_neurons=num_neurons,path=path).to(device)



class Distilled_GELU(nn.Module):
    def __init__(self, load_distilled_weights=False,num_neurons=64, path = None):
        super().__init__()
        self.approximator = nn.Sequential(
            nn.Linear(1, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, 1)
            )
        self.approximator.requires_grad = False
        
        if load_distilled_weights:
            self.load_state_dict(torch.load(path))
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        dim = x.dim()
        if dim == 0:
            return self.approximator(x)
        else:
            return torch.squeeze(self.approximator(torch.unsqueeze(x, -1)))
