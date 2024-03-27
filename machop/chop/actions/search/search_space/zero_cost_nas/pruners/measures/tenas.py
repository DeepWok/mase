'''
Author: ViolinSolo
Date: 2023-04-28 17:26:50
LastEditTime: 2023-05-06 10:22:37
LastEditors: ViolinSolo
Description: TENAS score computation
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/tenas.py
'''
'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/VITA-Group/TENAS
------------------------------------------------------------
modified by ViolinSolo from 
https://github.com/idstcv/ZenNAS/blob/main/ZeroShotProxy/compute_te_nas_score.py
'''

# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
import time
import torch
from torch import nn
import numpy as np
# import global_utils, argparse, ModelLoader, time
from . import measure

# =============================================================================
#  TENAS score computation: lrn part
# =============================================================================

class LinearRegionCount(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_samples, gpu=None):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None
        self.gpu = gpu


    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron)
            if self.gpu is not None:
                self.activations = self.activations.cuda(self.gpu)
        self.activations[self.ptr:self.ptr+n_batch] = torch.sign(activations)  # after ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        # #debug lines, try to check which device this variable is on
        # print('activations shape: ', self.activations.shape)
        # print('activations type: ', self.activations.dtype)
        # print('activations device: ', self.activations.device)

        # each element in res: A * (1 - B)
        res = torch.matmul(self.activations.half(), (1-self.activations).T.half())
        
        # make symmetric, each element in res: A * (1 - B) + (1 - A) * B, a non-zero element indicate a pair of two different linear regions
        # res += res.T # remove this line, replace inplace ops with non-inplace ops
        # # Bug Fix: RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.
        # # just avoid inplace operation.
        res = res + res.T

        # a non-zero element now indicate two linear regions are identical
        res = 1 - torch.sign(res)
        # for each sample's linear region: how many identical regions from other samples
        res = res.sum(1)
        # contribution of each redudant (repeated) linear region
        res = 1. / res.float()
        # sum of unique regions (by aggregating contribution of all regions)
        self.n_LR = res.sum().item()
        del self.activations, res
        self.activations = None
        if self.gpu is not None:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR

class Linear_Region_Collector:
    def __init__(self, models=[], input_size=(64, 3, 32, 32), gpu=None,
                 sample_batch=1, dataset=None, data_path=None, seed=0):
        self.models = []
        self.input_size = input_size  # BCHW
        self.sample_batch = sample_batch
        # self.input_numel = reduce(mul, self.input_size, 1)
        self.interFeature = []
        self.dataset = dataset
        self.data_path = data_path
        self.seed = seed
        self.gpu = gpu
        self.device = torch.device('cuda:{}'.format(self.gpu)) if self.gpu is not None else torch.device('cpu')
        # print('Using device:{}'.format(self.device))

        self.reinit(models, input_size, sample_batch, seed)


    def reinit(self, models=None, input_size=None, sample_batch=None, seed=None):
        if models is not None:
            assert isinstance(models, list)
            del self.models
            self.models = models
            for model in self.models:
                self.register_hook(model)
            self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch, gpu=self.gpu) for _ in range(len(models))]
        if input_size is not None or sample_batch is not None:
            if input_size is not None:
                self.input_size = input_size  # BCHW
                # self.input_numel = reduce(mul, self.input_size, 1)
            if sample_batch is not None:
                self.sample_batch = sample_batch
            # if self.data_path is not None:
            #     self.train_data, _, class_num = get_datasets(self.dataset, self.data_path, self.input_size, -1)
            #     self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.input_size[0], num_workers=16, pin_memory=True, drop_last=True, shuffle=True)
            #     self.loader = iter(self.train_loader)
        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            if self.gpu is not None:
                torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        if self.gpu is not None:
            torch.cuda.empty_cache()

    def clear(self):
        self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(self.models))]
        del self.interFeature
        self.interFeature = []
        if self.gpu is not None:
            torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())  # for ReLU

    def forward_batch_sample(self):
        for _ in range(self.sample_batch):
            # try:
            #     inputs, targets = self.loader.next()
            # except Exception:
            #     del self.loader
            #     self.loader = iter(self.train_loader)
            #     inputs, targets = self.loader.next()
            inputs = torch.randn(self.input_size, device=self.device)  ## TODO: change to real data? I don't if it is necessary?

            for model, LRCount in zip(self.models, self.LRCounts):
                self.forward(model, LRCount, inputs)
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        with torch.no_grad():
            # model.forward(input_data.cuda())
            model.forward(input_data)
            if len(self.interFeature) == 0: return
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)


@measure('lrn', bn=True, num_batch=1)
def compute_RN_score(net: nn.Module, inputs, targets, split_data=1, loss_fn=None,  # these are necessary arguments limited by *zero_cost_metrics.__init__.calc_metric*, if you want to add more arguments, modify @metric decorator's parameters to provide dynamic default values.
                     num_batch=None): # additional arguments
# def compute_RN_score(model: nn.Module,  batch_size=None, image_size=None, num_batch=None, gpu=None):
    # # # just debug
    # # gpu = 0
    # # import ModelLoader
    # # model = ModelLoader._get_model_(arch='resnet18', num_classes=1000, pretrained=False, opt=None, argv=None)
    # #
    # # if gpu is not None:
    # #     model = model.cuda(gpu)
    # lrc_model = Linear_Region_Collector(models=[model], input_size=(batch_size, 3, image_size, image_size),
    #                                     gpu=gpu, sample_batch=num_batch)
    
    device = inputs.device
    gpu = device.index
    # gpu = torch.cuda.current_device() #is same with former line
    lrc_model = Linear_Region_Collector(models=[net], input_size=tuple(inputs.size()),
                                        gpu=gpu, sample_batch=num_batch)
    # num_linear_regions = float(lrc_model.forward_batch_sample()[0])  # ViolinSolo: for standardization, we manually convert it to numpy.float64, matching the type of NTK score
    num_linear_regions = np.float64(lrc_model.forward_batch_sample()[0])
    del lrc_model
    torch.cuda.empty_cache()
    return num_linear_regions



# =============================================================================
#  TENAS score computation: ntk part
# =============================================================================


# import numpy as np
# import torch


def recal_bn(network, xloader, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network


def get_ntk_n(networks, recalbn=0, train_mode=False, num_batch=None,
              batch_size=None, image_size=None, gpu=None):
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]

    # for i, (inputs, targets) in enumerate(xloader):
    #     if num_batch > 0 and i >= num_batch: break
    for i in range(num_batch):
        inputs = torch.randn((batch_size, 3, image_size, image_size), device=device)
        # inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            if gpu is not None:
                inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            else:
                inputs_ = inputs.clone()

            logit = network(inputs_)
            if isinstance(logit, tuple):
                # logit = logit[1]  # 201 networks: return features and logits
                _, logit = logit  # 201 networks: return features and logits | same as above
                # ViolinSolo: just like other metrics, we assume if logit is tuple,
                #   the first element of logits is *feature*, the second is *logit*
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                if gpu is not None:
                    torch.cuda.empty_cache()

    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        # eigenvalues, _ = torch.symeig(ntk)  # ascending
        """ using following code to replace torch.symeig, because of the following warning:
        UserWarning: torch.symeig is deprecated in favor of torch.linalg.eigh and will be removed in a future PyTorch release.
        The default behavior has changed from using the upper triangular portion of the matrix by default to using the lower triangular portion.
        L, _ = torch.symeig(A, upper=upper)
        should be replaced with
        L = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')
        and
        L, V = torch.symeig(A, eigenvectors=True)
        should be replaced with
        L, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L')
        """
        eigenvalues = torch.linalg.eigvalsh(ntk, UPLO='U') # ascending
        # conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True))
    return conds


@measure('ntk', bn=True, num_batch=1)
def compute_NTK_score(net: nn.Module, inputs, targets, split_data=1, loss_fn=None,  # these are necessary arguments limited by *zero_cost_metrics.__init__.calc_metric*, if you want to add more arguments, modify @metric decorator's parameters to provide dynamic default values.
                     num_batch=None): # additional arguments
# def compute_NTK_score(gpu, model, resolution, batch_size):
    device = inputs.device
    gpu = device.index

    # (batch_size, 3, resolution, resolution) = inputs.shape
    batch_size, _, resolution, _ = inputs.size()

    ntk_score = get_ntk_n([net], recalbn=0, train_mode=True, num_batch=num_batch,
                           batch_size=batch_size, image_size=resolution, gpu=gpu)[0]
    return -1 * ntk_score


# =============================================================================
#  TENAS score computation: tenas part
# =============================================================================

@measure('tenas', bn=True, num_batch=1, bn_shadow=True)
def compute_TENAS_score(net: nn.Module, inputs, targets, split_data=1, loss_fn=None,  # these are necessary arguments limited by *zero_cost_metrics.__init__.calc_metric*, if you want to add more arguments, modify @metric decorator's parameters to provide dynamic default values.
                     num_batch=None, bn_shadow=True): # additional arguments
    net1 = net.get_prunable_copy(bn=bn_shadow)  # manually keep bn in lnwot, and remove bn in synflow
    ntk = compute_NTK_score(net1, inputs, targets, split_data, loss_fn, num_batch)
    del net1
    torch.cuda.empty_cache()
    gc.collect()

    net2 = net.get_prunable_copy(bn=bn_shadow)  # manually keep bn in lnwot, and remove bn in synflow
    RN = compute_RN_score(net2, inputs, targets, split_data, loss_fn, num_batch)
    del net2
    torch.cuda.empty_cache()
    gc.collect()

    the_score = ntk + RN
    return the_score


# def parse_cmd_options(argv):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
#     parser.add_argument('--input_image_size', type=int, default=None,
#                         help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
#     parser.add_argument('--repeat_times', type=int, default=32)
#     parser.add_argument('--gpu', type=int, default=None)
#     module_opt, _ = parser.parse_known_args(argv)
#     return module_opt



# if __name__ == "__main__":
#     opt = global_utils.parse_cmd_options(sys.argv)
#     args = parse_cmd_options(sys.argv)
#     the_model = ModelLoader.get_model(opt, sys.argv)
#     if args.gpu is not None:
#         the_model = the_model.cuda(args.gpu)


#     start_timer = time.time()

#     for repeat_count in range(args.repeat_times):
#         ntk = compute_NTK_score(gpu=args.gpu, model=the_model,
#                              resolution=args.input_image_size, batch_size=args.batch_size)
#         RN = compute_RN_score(model=the_model, batch_size=args.batch_size, image_size=args.input_image_size,
#                               num_batch=1, gpu=args.gpu)
#         the_score = RN + ntk
#     time_cost = (time.time() - start_timer) / args.repeat_times

#     print(f'ntk={the_score:.4g}, time cost={time_cost:.4g} second(s)')