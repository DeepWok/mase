'''
Author: ViolinSolo
Date: 2023-04-26 12:23:01
LastEditTime: 2023-04-28 21:44:16
LastEditors: ViolinSolo
Description: original RELU based NASWOT implementation.
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/naswot_relu.py
'''
import torch
import torch.nn as nn
import numpy as np
from . import measure


def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def safe_hooklogdet(K):
    s, ld = np.linalg.slogdet(K)
    return 0 if (np.isneginf(ld) and s==0) else ld

@measure("naswot_relu", bn=True)
@measure("nwot_relu_Kmats", bn=True, return_Kmats=True)
def compute_naswot(net, inputs, targets, loss_fn, split_data=1, 
                    return_Kmats=False):
    """
    This is the original RELU based NASWOT implementation.
    Based on v2 paper, and its repo link: 
    https://github.com/BayesWatch/nas-without-training/blob/master/score_networks.py
    """
    net.eval()


    net.K = 0. # **naswot matrix**, NONE-layer-wise | e (mat,) ===> using torch broadcasting tech to init zero-like matrix
    def counting_forward_hook(module, inp, out):
        # try:
        #    if not module.visited_backwards:
        #        return
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()  # remove this original code, since we want use broadcasting tech
        # except:
        #     pass


    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True


    for name, module in net.named_modules():
        if 'ReLU' in str(type(module)):
            #hooks[name] = module.register_forward_hook(counting_hook)
            module.register_forward_hook(counting_forward_hook)
            # module.register_backward_hook(counting_backward_hook)


    with torch.no_grad():
        net(inputs)

    K_mat = net.K
    K_mat_logdet = safe_hooklogdet(K_mat)
    return (K_mat, K_mat_logdet) if return_Kmats else K_mat_logdet