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
import json


ini_args = {
"model_name": "vit_base_patch16",
"dataset_name": "imagenet",
"batch_size": 32,
"load_name": None,
"load_type": None,
}
device='cuda:0'
mg, info = initialize_graph(**ini_args)
# for n in mg.fx_graph.nodes:
#     print(n.name)
acc = acc_cal(mg.model, info['data_module'].test_dataloader())
# a = 1

