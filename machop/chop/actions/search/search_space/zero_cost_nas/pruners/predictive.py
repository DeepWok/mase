# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .p_utils import *
from . import measures

import types
import copy

'''
group 2: zero_cost
This file is responsible for computing zero-cost metrics for neural network architectures. 

Functions:
1. **no_op**: 
    A placeholder function that returns its input without any modification.

2. **copynet**: 
    Creates a deep copy of a given network. If the `bn` parameter is False, it replaces the forward method of all BatchNorm layers in the network with the `no_op` function.

3. **find_measures_arrays**: 
    Computes the zero-cost metrics for a given network, data loader, and loss function. It first checks if the network has a `get_prunable_copy` method, and if not, it adds one. Then it moves the network to the CPU to free up GPU memory. It gets a minibatch of data based on the `dataload` parameter. It then tries to compute the zero-cost metrics for the network. If it runs out of memory, it splits the data into smaller parts and tries again.

4. **find_measures**: 
    A wrapper for the `find_measures_arrays` function. It computes the zero-cost metrics for a given network, data loader, and loss function. If the `measures_arr` parameter is not None, it uses the provided measures instead of computing them. It then sums up the measures and returns them.
'''

def no_op(self,x):
    return x

def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn==False:
        for l in net.modules():
            if isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d) :
                l.forward = types.MethodType(no_op, l)
    return net

def find_measures_arrays(net_orig, trainloader, dataload_info, device, measure_names=None, loss_fn=F.cross_entropy):
    # Group 2: Zero cost
    # This function computes the zero-cost metrics for a given network, data loader, and loss function. It first checks if the network has a `get_prunable_copy` method, and if not, it adds one. Then it moves the network to the CPU to free up GPU memory. It gets a minibatch of data based on the `dataload` parameter. It then tries to compute the zero-cost metrics for the network. If it runs out of memory, it splits the data into smaller parts and tries again.

    if measure_names is None:
        measure_names = measures.available_measures

    dataload, num_imgs_or_batches, num_classes = dataload_info

    if not hasattr(net_orig,'get_prunable_copy'):
        net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

    #move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu() 
    torch.cuda.empty_cache()

    #given 1 minibatch of data
    if dataload == 'random':
        inputs, targets = get_some_data(trainloader, num_batches=num_imgs_or_batches, device=device)
    elif dataload == 'grasp':
        inputs, targets = get_some_data_grasp(trainloader, num_classes, samples_per_class=num_imgs_or_batches, device=device)
    else:
        raise NotImplementedError(f'dataload {dataload} is not supported')

    done, ds = False, 1
    measure_values = {}

    while not done:
        try:
            for measure_name in measure_names:
                if measure_name not in measure_values:
                    val = measures.calc_measure(measure_name, net_orig, device, inputs, targets, loss_fn=loss_fn, split_data=ds)
                    measure_values[measure_name] = val

            done = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                done=False
                if ds == inputs.shape[0]//2:
                    raise ValueError(f'Can\'t split data anymore, but still unable to run. Something is wrong') 
                ds += 1
                while inputs.shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f'Caught CUDA OOM, retrying with data split into {ds} parts')
            else:
                raise e

    net_orig = net_orig.to(device).train()
    return measure_values

def find_measures(net_orig,                  # neural network
                  dataloader,                # a data loader (typically for training data)
                  dataload_info,             # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
                  device,                    # GPU/CPU device used
                  loss_fn=F.cross_entropy,   # loss function to use within the zero-cost metrics
                  measure_names=None,        # an array of measure names to compute, if left blank, all measures are computed by default
                  measures_arr=None):        # [not used] if the measures are already computed but need to be summarized, pass them here
    # Group 2: Zero cost
    # This function is a wrapper for the `find_measures_arrays` function. It computes the zero-cost metrics for a given network, data loader, and loss function. If the `measures_arr` parameter is not None, it uses the provided measures instead of computing them. It then sums up the measures and returns them.
    #Given a neural net
    #and some information about the input data (dataloader)
    #and loss function (loss_fn)
    #this function returns an array of zero-cost proxy metrics.

    def sum_arr(arr):
        sum = 0.
        if hasattr(arr, '__len__'):
            for i in range(len(arr)):
                val = arr[i]
                val = val if isinstance(val, torch.Tensor) else torch.tensor(val)
                sum += torch.sum(arr[i])
        else:
            sum = arr
        return sum.item() if hasattr(sum, 'item') else sum

    if measures_arr is None:
        measures_arr = find_measures_arrays(net_orig, dataloader, dataload_info, device, loss_fn=loss_fn, measure_names=measure_names)

    measures = {}
    for k,v in measures_arr.items():
        if k=='jacob_cov':
            measures[k] = v
        else:
            measures[k] = sum_arr(v)

    return measures
