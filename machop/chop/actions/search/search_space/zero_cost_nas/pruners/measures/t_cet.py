"""
Author: ViolinSolo
Date: 2023-04-29 17:33:33
LastEditTime: 2023-05-05 17:10:54
LastEditors: ViolinSolo
Description: t_cet metric
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/t_cet.py
"""

import gc
import numpy as np
import torch

from . import measure
from .naswot import compute_naswot
from .synflow import compute_synflow_per_weight
from .snip import compute_snip_per_weight


@measure("tcet_syn_none", bn=True, mode="none", mt="synflow")  # change bn from False to True, in order to avoid lnwot changes. We will manually remove bn in compute_tcet_score set synflow WITHOUT bn.
@measure("tcet_syn_log", bn=True, mode="log", mt="synflow")  # change bn from False to True, in order to avoid lnwot changes. We will manually remove bn in compute_tcet_score set synflow WITHOUT bn.
@measure("tcet_syn_log1p", bn=True, mode="log1p", mt="synflow")  # change bn from False to True, in order to avoid lnwot changes. We will manually remove bn in compute_tcet_score set synflow WITHOUT bn.
@measure("tcet_syn_norm", bn=True, mode="norm", mt="synflow")  # change bn from False to True, in order to avoid lnwot changes. We will manually remove bn in compute_tcet_score set synflow WITHOUT bn.
@measure("tcet_snip_none", bn=True, mode="none", mt="snip")
@measure("tcet_snip_log", bn=True, mode="log", mt="snip")
@measure("tcet_snip_log1p", bn=True, mode="log1p", mt="snip")
@measure("tcet_snip_norm", bn=True, mode="norm", mt="snip")
@measure("t_cet", bn=True, mode="none", mt="synflow")  # change bn from False to True, in order to avoid lnwot changes. We will manually remove bn in compute_tcet_score set synflow WITHOUT bn.
def compute_tcet_score(net, inputs, targets, loss_fn=None, split_data=1,
                        mode="none", mt="synflow"):
    
    net1 = net.get_prunable_copy(bn=True)  # manually keep bn in lnwot, and remove bn in synflow
    lnwot_scores = compute_naswot(net1, inputs, targets, loss_fn, split_data, layerwise=True, return_Kmats=False)
    del net1
    torch.cuda.empty_cache()
    gc.collect()

    net2 = net.get_prunable_copy(bn=True if mt == "snip" else False)  # manually keep bn in lnwot, and remove bn in synflow
    mt_scores = compute_snr_score(net2, inputs, targets, loss_fn, split_data, mode=mode, mt=mt)
    del net2
    torch.cuda.empty_cache()
    gc.collect()

    tcet_scores = []
    for lnwot_score, mt_score in zip(lnwot_scores, mt_scores):
        tcet_scores.append(lnwot_score * mt_score)
    return tcet_scores



@measure("snr_syn_none", bn=False, mode="none", mt="synflow")
@measure("snr_syn_log", bn=False, mode="log", mt="synflow")
@measure("snr_syn_log1p", bn=False, mode="log1p", mt="synflow")
@measure("snr_syn_norm", bn=False, mode="norm", mt="synflow")
@measure("snr_snip_none", bn=True, mode="none", mt="snip")
@measure("snr_snip_log", bn=True, mode="log", mt="snip")
@measure("snr_snip_log1p", bn=True, mode="log1p", mt="snip")
@measure("snr_snip_norm", bn=True, mode="norm", mt="snip")
@measure("snr_syn", bn=False, mode="none", mt="synflow")
@measure("snr_snip", bn=True, mode="none", mt="snip")
def compute_snr_score(net, inputs, targets, loss_fn=None, split_data=1,
                        mode="none", mt="synflow"):
    
    if mt == "synflow":
        synflow_scores = compute_synflow_per_weight(net, inputs, targets, mode="param", split_data=split_data, loss_fn=loss_fn)
        mt_scores = synflow_scores
    elif mt == "snip":
        snip_scores = compute_snip_per_weight(net, inputs, targets, mode="param", split_data=split_data, loss_fn=loss_fn)
        mt_scores = snip_scores
    
    snr_mt_scores = []
    for mt_score in mt_scores:
        s = mt_score.detach().view(-1)

        sigma = torch.std(s)
        sigma = torch.tensor(0.) if torch.isnan(sigma) else sigma
        
        if sigma == 0 or torch.isnan(sigma):
            s = torch.sum(s)
        else:
            s = torch.sum(s)/sigma
            
        s_val = s.cpu().item()
        snr_mt_scores.append(s_val)

    if mode == "none":
        return snr_mt_scores
    elif mode == "log":
        log_sc = np.log(snr_mt_scores)
        return np.nan_to_num(log_sc, nan=0.0, posinf=0.0, neginf=0.0)
    elif mode == "log1p":
        log_sc = np.log1p(snr_mt_scores)
        return np.nan_to_num(log_sc, nan=0.0, posinf=0.0, neginf=0.0)
    elif mode == "norm":
        _mt_std = np.std(snr_mt_scores)
        return [s/_mt_std if _mt_std>0 else s for s in snr_mt_scores]
