import logging
import os
from os import PathLike
import toml
import torch
import numpy as np
import json

from fvcore.common.config import CfgNode
from ...tools.checkpoint_load import load_model
from ...tools.config_load import load_config
from ...tools.get_input import get_dummy_input
# from .search_space import get_search_space_cls
# from .strategies import get_search_strategy_cls
from chop.tools.utils import device
from chop.tools.utils import parse_accelerator


from naslib.utils import get_zc_benchmark_api,get_dataset_api
from naslib.utils import get_train_val_loaders, get_project_root
from naslib.search_spaces import NasBench101SearchSpace , NasBench201SearchSpace , NasBench301SearchSpace
from naslib.predictors import ZeroCost
from naslib.search_spaces.core import Metric


# For training meta-proxy network
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from einops import rearrange
# from torch import optim
# import torch.nn.functional as F
# import time


logger = logging.getLogger(__name__)

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def parse_proxy_config(config):
    try:
        proxy_config = config['proxy']
        op_config = proxy_config['op_config']
        proxies = proxy_config['proxies']
        dataset_info = proxy_config['proxy_dataset']
        search_space_info = proxy_config['search_space']

        return op_config['num_samples'], proxies['proxy_list'], dataset_info['dataset'], search_space_info['search_space']
    
    except:
        logger.info("Invalid Config!")
        exit()

    # try:
    #     search_config = config["proxy"]   #p arse into search config
    #     nas_config = search_config['nas']  # parse into nas
    #     op_config = nas_config['op_config']
    #     proxy_config = nas_config['proxy_config']
    #     dataset_info = nas_config['proxy_dataset']
    #     search_space_info = nas_config['search_space']
    #     if search_space_info['search_space'] not in ["nas201", "nas301"]:
    #         logger.info("Invalid Search Space!")
    #         exit()
    #     return op_config['op_indices'], proxy_config['proxy'], dataset_info['dataset'], search_space_info['search_space'] # one more time one more chance running

def proxy(config:dict | PathLike):

    if not isinstance(config, dict):
        config = load_config(config)
    op_config, proxy_config, dataset_info, search_space_info = parse_proxy_config(config)   #op_config = list of integers , proxy_config = list of strings

    dataset_info = "cifar10"
    # accepted_dataset = ["cifar10", "cifar100", "ImageNet16-120"]
    # if search_space_info == "nas101" or search_space_info == "nas301":
    #     dataset_info = "cifar10"
    # elif search_space_info == "nas201":
    #     if dataset_info not in accepted_dataset:
    #         dataset_info = "cifar10"

    switch = {
    "cifar10": 10,
    "cifar100": 100,
    "ImageNet16-120": 120
    }

    n_classes = switch[dataset_info]

    ### Prepare dataloader for running proxies
    config_dict = {
        'dataset': dataset_info, # Dataset to loader: can be cifar10, cifar100, ImageNet16-120
        'data': str(get_project_root()) + '/data', # path to naslib/data where cifar is saved
        'search': {
            'seed': 9001, # Seed to use in the train, validation and test dataloaders
            'train_portion': 0.7, # Portion of train dataset to use as train dataset. The rest is used as validation dataset.
            'batch_size': 32, # batch size of the dataloaders
        }
    }
    
    dataset_config = CfgNode(config_dict)
    train_loader, val_loader, test_loader, train_transform, valid_transform = get_train_val_loaders(dataset_config)
    scores = {}
#   
    while len(list(scores.keys())) < op_config:
        # Generate models
        # if search_space_info == "nas101":
        #     graph = NasBench101SearchSpace(n_classes)
        if search_space_info == "nas201":
            graph = NasBench201SearchSpace(n_classes)
        elif search_space_info == "nas301":
            graph = NasBench301SearchSpace(n_classes)
        graph.sample_random_architecture(None)
        graph.parse()
        op = graph.get_hash()
        if str(op) not in scores:
            scores[str(op)]={}
            for zc_proxy in proxy_config:
                zc_predictor = ZeroCost(method_type=zc_proxy)
                score = zc_predictor.query(graph=graph, dataloader = train_loader)
                scores[str(op)][zc_proxy] = score

    # Path for saving the scores
    file_path = "../nas_results/proxy_scores.json"
    # Write dictionary to JSON file
    with open(file_path, 'w') as json_file:
        json.dump(scores, json_file)


    # Calculate stddev and mean for future data normalisation
    proxy_mean_stddev = {}
    for zc_proxy in proxy_config:
        temp = []
        proxy_mean_stddev[zc_proxy] = {}
        for key in scores:
            score = scores[key][zc_proxy] 
            temp.append(score)
        temp = np.array(temp)
        mean = np.mean(temp)
        stddev = np.std(temp)
        if stddev == 0:
            stddev = 1e-8
        proxy_mean_stddev[zc_proxy]['mean'] = mean
        proxy_mean_stddev[zc_proxy]['stddev'] = stddev

    # Save mean and standard deviation of proxy score distribution
    file_path = "../nas_results/proxy_mean_stddev.json"
    # Write dictionary to JSON file
    with open(file_path, 'w') as json_file:
        json.dump(proxy_mean_stddev, json_file)
    return

# dataset_infoCreate list of indecies for architectures to be quired in nas-bench
# indicies_list = []
# while len(indicies_list) < op_config:
#     small_rand_list = [int(np.random.rand()*5) for _ in range(6)]
#     if small_rand_list not in indicies_list:
#         indicies_list.append(small_rand_list)

# Prepare list and dict for recording scores
# # Prepare dataset for training meta-proxy
# proxy = proxy_config
# data_set = []
# for proxy_name in proxy:
#     data_set.append(np.array(scores[proxy_name]))
# data_set = np.array(data_set)
# features = rearrange(data_set, 'a b -> b a')


# def predictor_trian(dataloader, proxy):
#     class NeuralModel(nn.Module):
#         def __init__(self, input_size):
#             super(NeuralModel, self).__init__()
#             self.linear1 = nn.Linear(input_size, 64)
#             self.sigmoid = nn.Sigmoid()
#             self.linear2 = nn.Linear(64, 128)
#             self.relu = nn.ReLU()
#             self.linear3 = nn.Linear(128, 1)

#         def forward(self, x):
#             x = self.sigmoid(self.linear1(x))
#             x = self.relu(self.linear2(x))
#             x = torch.sigmoid(self.linear3(x))
#             return x
    
#     input_size = len(proxy)
#     model = NeuralModel(input_size)
#     criterion = nn.L1Loss()
#     optimizer = optim.Adam(model.parameters(), lr = 0.01)
    
#     # Train it    
#     num_epochs = 1000
#     for epoch in range(num_epochs):
#         for inputs, targets in dataloader:
#             # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs.squeeze(), targets)
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         if (epoch+1) % 50 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#     # Saved the trained model, this is the daddy proxy
#     relative_path = r'nas_results/model_state_dict.pt'
#     current_directory = os.getcwd()
#     one_level_up_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
#     file_path = os.path.join(one_level_up_directory, relative_path)
#     torch.save(model.state_dict(), file_path)
#     return


    # labels = np.array(val_accuries)
    # labels /= 100  # Convert accuracy into a float between 0 to 1
    # features = torch.Tensor(features)
    # labels = torch.Tensor(labels)
    # dataset = TensorDataset(features, labels)                       # Prepare it for pytorch format
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)    # Prepare for pytorch format



    # train_acc_parent = graph.query(metric=Metric.TRAIN_ACCURACY, dataset='cifar10', dataset_api=dataset_apis["NASBench201-cifar10"])
    # val_acc_parent = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=dataset_apis["NASBench201-cifar10"])
            # train_accuries.append(train_acc_parent)
    # val_accuries.append(val_acc_parent)
    
    # nas201_dataset_names = ["cifar10","cifar100","ImageNet16-120"]
    #     # Read json file that contain proxy score and validation accuracy of each architecture in search space 
    #     file_path = './naslib/data/zc_nasbench201.json'
    # if dataset_info in nas201_dataset_names:                #Ensure the dataset chosen is in nasbench201
    #     nas201_record =  read_json_file(file_path)
    #     data = nas201_record[dataset_info] 
        
    #     indicies_list = []
    #     while len(indicies_list) < op_config:
    #         small_rand_list = [int(np.random.rand()*5) for _ in range(6)]
    #         if small_rand_list not in indicies_list:
    #             indicies_list.append(small_rand_list)
    #     scores = {}
    #     val_accuries = []
    #     for proxy_name in proxy_config:
    #         scores[proxy_name] = []
    #     for op in indicies_list:
    #         op= '(' + ', '.join(map(str, op)) + ')'
    #         proxy_score = data[op]
    #         for proxy_name in proxy_config:
    #             scores[proxy_name].append(proxy_score[proxy_name]['score'])
    #         val_accuries.append(proxy_score['val_accuracy'])

    #     # Prepare dataset for training meta-proxy
    #     proxy = proxy_config
    #     data_set = []
    #     for proxy_name in proxy:
    #         data_set.append(np.array(scores[proxy_name]))
    #     data_set = np.array(data_set)
    #     features = rearrange(data_set, 'a b -> b a')
    #     labels = np.array(val_accuries)
    #     labels /= 100
    #     features = torch.Tensor(features)
    #     labels = torch.Tensor(labels)
    #     dataset = TensorDataset(features, labels)                       # Prepare it for pytorch format
    #     dataloader = DataLoader(dataset, batch_size=8, shuffle=True)    # Prepare for pytorch format
    #     predictor_trian(dataloader,proxy)
    # elif dataset_info=='ptb':
    #     pass
    # else:
    #     print("The dataset config is not available")
    #     exit()