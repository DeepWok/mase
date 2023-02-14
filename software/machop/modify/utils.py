import torch
import numpy


use_cuda = torch.cuda.is_available()
torch_cuda = torch.cuda if use_cuda else torch
device = torch.device('cuda:0' if use_cuda else 'cpu')

def to_numpy(x):
    if use_cuda:
        x = x.cpu()
    return x.detach().numpy()

def to_tensor(x):
    return torch.from_numpy(x).to(device)


