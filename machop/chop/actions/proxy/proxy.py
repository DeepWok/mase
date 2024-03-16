import logging
from os import PathLike

import toml
import torch

from fvcore.common.config import CfgNode
from ...tools.checkpoint_load import load_model
from ...tools.config_load import load_config
from ...tools.get_input import get_dummy_input
from .search_space import get_search_space_cls
from .strategies import get_search_strategy_cls
from chop.tools.utils import device
from chop.tools.utils import parse_accelerator

import numpy as np
from .naslib.search_spaces import NasBench201SearchSpace
from .naslib.utils import get_zc_benchmark_api,get_dataset_api
from .naslib.utils import get_train_val_loaders, get_project_root
from .naslib.predictors import ZeroCost
from .naslib.search_spaces.core import Metric

# For training  meta-proxy network
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from einops import rearrange
from torch import optim
import torch.nn.functional as F

def parse_nas_config(config):
    search_config = config["search"]   #p arse into search config
    nas_config = search_config['nas']  # parse into nas
    op_config = nas_config['op_config']
    proxy_config = nas_config['proxy_config']
    return op_config['op_indices'], proxy_config['proxy']# one more time one more chance running
    
def proxy_predcitor(config):
    op_config, proxy_config = parse_nas_config(config)   # type(op_config) = list of integers , type(proxy_config) = list of strings
        
    # Create list of indecies for architectures to be quired in nas-bench
    indicies_list = []
    while len(indicies_list) < op_config:
        small_rand_list = [int(np.random.rand()*5) for _ in range(6)]
        if small_rand_list not in indicies_list:
            indicies_list.append(small_rand_list)
    

    # Prepare list and dict for recording scores
    scores = {}
    for proxy_name in proxy_config:
        scores[proxy_name] = []
    train_accuries = []
    val_accuries = []
    config_dict = {
        'dataset': 'cifar10', # Dataset to loader: can be cifar10, cifar100, ImageNet16-120
        'data': str(get_project_root()) + '/data', # path to naslib/data where cifar is saved
        'search': {
            'seed': 9001, # Seed to use in the train, validation and test dataloaders
            'train_portion': 0.7, # Portion of train dataset to use as train dataset. The rest is used as validation dataset.
            'batch_size': 32, # batch size of the dataloaders
        }
    }
    config = CfgNode(config_dict)
    train_loader, val_loader, test_loader, train_transform, valid_transform = get_train_val_loaders(config)
    dataset_apis={}
    dataset_apis["NASBench201-cifar10"] = get_dataset_api(search_space='nasbench201', dataset='cifar10')

    for op in indicies_list:
        # config_dict is config from nas-bench
        # Generate models
        graph = NasBench201SearchSpace(n_classes=10)
        graph.sample_architecture(op_indices=op)
        # graph.sample_random_architecture()
        graph.parse()
        graph.get_hash()
        train_acc_parent = graph.query(metric=Metric.TRAIN_ACCURACY, dataset='cifar10', dataset_api=dataset_apis["NASBench201-cifar10"])
        val_acc_parent = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=dataset_apis["NASBench201-cifar10"])
        
        train_accuries.append(train_acc_parent)
        val_accuries.append(val_acc_parent)


        for zc_proxy in proxy_config:
            zc_predictor = ZeroCost(method_type=zc_proxy)
            score = zc_predictor.query(graph=graph, dataloader=train_loader)
            # print(score)
            scores[zc_proxy].append(score)
    # np.save(r'/home/ansonhon/mase_project/nas_results/scores_test3',scores)
    # np.save(r'/home/ansonhon/mase_project/nas_results/train_acc_test3',train_accuries)
    # np.save(r'/home/ansonhon/mase_project/nas_results/val_acc_test3',val_accuries)

    ### TODO: Add Meta-proxy neural network training here


    # Prepare dataset for training meta-proxy
    proxy = proxy_config
    data_set = []
    for proxy_name in proxy:
        data_set.append(np.array(scores[proxy_name]))
    data_set = np.array(data_set)
    features = rearrange(data_set, 'a b -> b a')
    labels = np.array(val_accuries)
    labels /= 100
    features = torch.Tensor(features)
    labels = torch.Tensor(labels)
    dataset = TensorDataset(features, labels)                       # Prepare it for pytorch format
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)    # Prepare for pytorch format

    # Define model to be trained as meta-proxy
    class NeuralModel(nn.Module):
        def __init__(self, input_size):
            super(NeuralModel, self).__init__()
            self.linear1 = nn.Linear(input_size, 64)
            self.sigmoid = nn.Sigmoid()
            self.linear2 = nn.Linear(64, 128)
            self.relu = nn.ReLU()
            self.linear3 = nn.Linear(128, 1)

        def forward(self, x):
            x = self.sigmoid(self.linear1(x))
            x = self.relu(self.linear2(x))
            x = torch.sigmoid(self.linear3(x))
            return x
    
    input_size = len(proxy)
    model = NeuralModel(input_size)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    
    # Train it    
    num_epochs = 1000
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Saved the trained model
    torch.save(model.state_dict(), '/home/ansonhon/mase_project/nas_results/model_state_dict.pt')

    return