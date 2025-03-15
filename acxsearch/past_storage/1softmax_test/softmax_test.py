import sys
import time
from pathlib import Path
import torch
import argparse
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1].as_posix()
sys.path.append(ROOT)
sys.path.append(ROOT + "/machop")

from search_func import _in_layer_quant_search

from machop.chop.passes.graph.analysis import add_software_metadata_analysis_pass
from machop.chop.passes.graph.transforms.quantize import (
    quantize_transform_pass,
    softmax_transform_pass,
)
from a4cirrus.utils import acc_cal, loss_cal, initialize_graph
from a4cirrus.search_func.utils import _get_similarity
import json

from machop.chop.passes.graph.transforms.quantize.quantizers.integer import integer_quantizer
from machop.chop.passes.graph.transforms.quantize.hash_modules import hash_softmax

import torch
from math import ceil, log2

x = 9 * torch.randn(1,64, 196)
config={
    'data_in_width':16,
    'data_in_frac_width':8,
    'data_in_exp_width':48,
    'data_in_exp_frac_width':8,
    'data_in_div_frac_width':7,
}
dim=-1
quant_x = integer_quantizer(x, config["data_in_width"], config["data_in_frac_width"])

print(_get_similarity(quant_x, x, "cosine").mean())
exp_x = quant_x.exp()
print(quant_x.abs().max())
quant_exp = integer_quantizer(exp_x, config["data_in_exp_width"], config["data_in_exp_frac_width"])

print(_get_similarity(quant_exp, exp_x, "cosine").mean())
exp_sum = quant_exp.sum(dim=dim, keepdim=True)
extra_sum_width = ceil(log2(x.shape[dim]))
after_div_frac_width = extra_sum_width - config["data_in_exp_frac_width"]
shift_width = config["data_in_div_frac_width"] - after_div_frac_width
if torch.all(quant_exp == exp_sum):
    out = torch.tensor(1.0, device=x.device).expand(x.shape)
else:
    out = quant_exp * (2 ** (shift_width)) // exp_sum
    out = out / (2 ** (shift_width))

print(_get_similarity(out, x.softmax(dim), "cosine").mean())
# hash division
# reci_out = IntHashFunc.apply(exp_sum, torch.reciprocal, config["exp_reciprocal"])
# x = x * reci_out
# hs_x = hash_softmax(x, -1, config)

# print(_get_similarity(hs_x, x.softmax(-1), "cosine").mean())

