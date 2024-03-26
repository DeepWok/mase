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
import types
import copy

from .p_utils import *
from . import measures
from .measures.model_stats import get_model_stats


def no_op(self, x):
    return x


def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn == False:
        for l in net.modules():
            if isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm1d):
                l.forward = types.MethodType(no_op, l)
    return net


# class CustomLoss(nn.Module):
#     def __init__(self, size=None,  weight=None, bias=None):
#         super(CustomLoss, self).__init__()
#         self.weight = weight
#         self.bias = bias
#         self.size = size

#     def forward(self, output, target, weight=None, bias=None):
#         if weight is None:
#             weight = self.weight
#         if bias is None:
#             bias = self.bias
#         criterion = SplitCrossEntropyLoss(self.size, splits=[], verbose=False)
#         loss = criterion(weight, bias, output, target)

#         return loss


def find_nlp_measures_arrays(
    net_orig,
    inputs,
    target,
    device,
    measure_names=None,
    loss_fn=F.cross_entropy,
):


    ds = 1
    measure_values = {}



    for measure_name in measure_names:
        if measure_name not in measure_values:
            val = measures.calc_measure(
                measure_name,
                net_orig,
                device,
                inputs,
                target,
                # target_onehot,
                loss_fn=loss_fn,
                split_data=ds,
            )
            measure_values[measure_name] = val

        # done = True
        # except RuntimeError as e:
        #     if "out of memory" in str(e):
        #         done = False
        #         if ds == inputs.shape[0] // 2:
        #             raise ValueError(
        #                 f"Can't split data anymore, but still unable to run. Something is wrong"
        #             )
        #         ds += 1
        #         while inputs.shape[0] % ds != 0:
        #             ds += 1
        #         torch.cuda.empty_cache()
        #         print(f"Caught CUDA OOM, retrying with data split into {ds} parts")
        #     else:
        #         raise e
    
    # net_orig = net_orig.to(device).train()
    return measure_values
  



def find_measures_arrays(
    net_orig,
    trainloader,
    dataload_info,
    device,
    measure_names=None,
    loss_fn=F.cross_entropy,
):
    device="cpu"
    if measure_names is None:
        measure_names = measures.available_measures

    dataload, num_imgs_or_batches, num_classes = dataload_info

    if not hasattr(net_orig, "get_prunable_copy"):
        net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)
    # move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu()
    torch.cuda.empty_cache()

    # given 1 minibatch of data
    if dataload == "random":
        inputs, targets = get_some_data(
            trainloader, num_batches=num_imgs_or_batches, device=device
        )
    elif dataload == "grasp":
        inputs, targets = get_some_data_grasp(
            trainloader,
            num_classes,
            samples_per_class=num_imgs_or_batches,
            device=device,
        )
    else:
        raise NotImplementedError(f"dataload {dataload} is not supported")

    done, ds = False, 1
    measure_values = {}

    # Turn target into one-hot vector
    inputs = inputs[0:512,:]   ###
    targets = targets[0:512]
    # target_onehot = F.one_hot(targets, num_classes = max(targets)+1)

    # # Convert datatype to float
    # inputs = torch.tensor(inputs, dtype = torch.float)
    # target_onehot = torch.tensor(target_onehot, dtype = torch.float)


    while not done:
        try:
            for measure_name in measure_names:
                
                if measure_name not in measure_values and measure_name != "flops" and measure_name != "params":
                    # import time
                    # start = time.time()
                    val = measures.calc_measure(
                        measure_name,
                        net_orig,
                        device,
                        inputs,
                        targets,
                        # target_onehot,
                        loss_fn=loss_fn,
                        split_data=ds,
                    )
                    # end = time.time()
                    # print(end - start)
                    measure_values[measure_name] = val

            done = True
        except RuntimeError as e:
            if "out of memory" in str(e):
                done = False
                if ds == inputs.shape[0] // 2:
                    raise ValueError(
                        f"Can't split data anymore, but still unable to run. Something is wrong"
                    )
                ds += 1
                while inputs.shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f"Caught CUDA OOM, retrying with data split into {ds} parts")
            else:
                raise e
    
    net_orig = net_orig.to(device).train()
    return measure_values


def find_measures(
    net_orig,  # neural network
    dataloader,  # a data loader (typically for training data)
    dataload_info,  # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
    device,  # GPU/CPU device used
    loss_function,  # loss function to use within the zero-cost metrics
    measure_names=None,  # an array of measure names to compute, if left blank, all measures are computed by default
    measures_arr=None,
    nlp = False
):

    def sum_arr(arr):
        sum = 0.0
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()

    measure_score={}
    data_iterator = iter(dataloader)
    x, target = next(data_iterator)
    x_shape = list(x.shape)
    x_shape[0] = 1 # to prevent overflow

    model_stats = get_model_stats(
        net_orig,
        input_tensor_shape=x_shape,
        clone_model=True
    )
    if 'flops' in measure_names:
        measure_score['flops'] = float(model_stats.Flops)/1e6
    if 'params' in measure_names:
        measure_score['params'] = float(model_stats.parameters)/1e6
        

    if measures_arr is None:

        measures_arr = find_measures_arrays(
            net_orig,
            dataloader,
            dataload_info,
            device,
            loss_fn=loss_function,
            measure_names=measure_names,
        )


    for k, v in measures_arr.items():
        if k == "jacov" or k == 'epe_nas' or k=='nwot' or k=='zen':
            measure_score[k] = v
        else:
            measure_score[k] = sum_arr(v)

    return measure_score




# def find_nlp_measures(
#     net_orig,  # neural network
#     inputs,
#     target,  # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
#     device,  # GPU/CPU device used
#     loss_function,  # loss function to use within the zero-cost metrics
#     emsize,
#     measure_names=None,  # an array of measure names to compute, if left blank, all measures are computed by default
#     measures_arr=None
# ):

#     # Given a neural net
#     # and some information about the input data (dataloader)
#     # and loss function (loss_fn)
#     # this function returns an array of zero-cost proxy metrics.

#     def sum_arr(arr):
#         sum = 0.0
#         for i in range(len(arr)):
#             sum += torch.sum(arr[i])
#         return sum.item()


#     measure_score={}
#     # data_iterator = iter(dataloader)

#     # x = next(data_iterator)
#     # x = np.array(x)
#     # x_shape = list(x.shape)
#     # x_shape[0] = 1 # to prevent overflow
#     # print("shape of x: ", x.shape)
#     # model_stats = get_model_stats(
#     #     net_orig,
#     #     input_tensor_shape=x.shape,
#     #     clone_model=True
#     # )

#     # if 'flops' in measure_names:
#     #     measure_score['flops'] = float(model_stats.Flops)/1e6
#     #     measure_names.remove('flops')
#     # if 'params' in measure_names:
#     #     measure_score['params'] = float(model_stats.parameters)/1e6
#     #     measure_names.remove('params')
        
#     # x = torch.rand(70, 50).to('cuda')

#     loss_function = CustomLoss(emsize, net_orig.decoder.weight, net_orig.decoder.bias)

#     if measures_arr is None:
#         measures_arr = find_nlp_measures_arrays(
#             net_orig,
#             inputs,
#             target,
#             device,
#             measure_names=measure_names,
#             loss_fn=loss_function
#         )

#     for k, v in measures_arr.items():
#         if k == "jacov" or k == 'epe_nas' or k=='nwot' or k=='zen':
#             measure_score[k] = v
#         else:
#             measure_score[k] = sum_arr(v)
#     return measure_score

    ######## Original NASLib calculation for getting flops and number of params in a model##########
    # if measure_names[0] in ['flops', 'params']:
        # data_iterator = iter(dataloader)
        # x, target = next(data_iterator)
        # x_shape = list(x.shape)
        # x_shape[0] = 1 # to prevent overflow

        # model_stats = get_model_stats(
        #     net_orig,
        #     input_tensor_shape=x_shape,
        #     clone_model=True
        # )

    #     if measure_names[0] == 'flops':
    #         measure_score = float(model_stats.Flops)/1e6 # megaflops
    #     else:
    #         measure_score = float(model_stats.parameters)/1e6 # megaparams
    #     return measure_score
    #################################################################################################