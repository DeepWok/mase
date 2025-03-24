import os, sys, logging, traceback, pdb

from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity
def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)
sys.excepthook = excepthook

logger = get_logger(__name__)
set_logging_verbosity("debug")

checkpoint = "resnet18"

from utils import init_dataset, acc_cal, fine_tune

import timm

from chop.dataset import get_dataset_info, MaseDataModule
from chop.models import get_model, get_model_info


import torch
batch_size = 128
model = timm.create_model(checkpoint, pretrained=True, num_classes=10)
datamodule = init_dataset("cifar10", batch_size, checkpoint)
from cim.module_level_tranform import vit_module_level_quantize
model.load_state_dict(torch.load("/home/cx922/mase/acxsearch/checkpoint/ckpt.pth"))
acc = acc_cal(model, datamodule)
print(acc)
breakpoint()
qmodel = vit_module_level_quantize(model, {
    "by": "type",
    "conv2d": {
        "config": {
            "num_bits": 8,
            "weight_range": 1.0,
            "bias_range": 1.0,
            "range_decay": 0.0,
            "noise_magnitude": 0.01,
        }
    },
    "linear": {
        "config": {
            "num_bits": 8,
            "weight_range": 1.0,
            "bias_range": 1.0,
            "range_decay": 0.0,
            "noise_magnitude": 0.01,
        }
    }
})
# model(next(iter(datamodule.train_dataloader()))[0])
# fine_tune(model, {
#     "model_info": get_model_info(checkpoint),
#     "task": "classification",
#     "dataset_info": get_dataset_info("cifar10"),
#     "data_module": datamodule,
#     "num_batchs": batch_size,
#     "epochs": 200,
# })



