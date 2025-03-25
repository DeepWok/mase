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
import timm
import torch.backends.cudnn as cudnn
net = timm.create_model("resnet18", pretrained=True, num_classes=10)
net = net.to("cuda")
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

# Load checkpoint.
print('==> Loading checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
# best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']
# acc = acc_cal(net, datamodule.test_dataloader())
# print(acc)
# breakpoint()
qmodel = vit_module_level_quantize(net, {
    "by": "type",
    "conv2d": {
        "config": {
            "num_bits": 8,
            "weight_range": 1.0,
            "bias_range": 1.0,
            "range_decay": 0.0,
            "noise_magnitude": 0.2,
        }
    },
    "linear": {
        "config": {
            "num_bits": 8,
            "weight_range": 1.0,
            "bias_range": 1.0,
            "range_decay": 0.0,
            "noise_magnitude": 0.2,
        }
    },
    "batch_norm": {
        "config": {
            "num_bits": 8,
            "max_value": 6.0,
            "decay": 1e-3,
        }
    }
})

acc = acc_cal(qmodel, datamodule.test_dataloader())
print(acc)
# model(next(iter(datamodule.train_dataloader()))[0])
# fine_tune(model, {
#     "model_info": get_model_info(checkpoint),
#     "task": "classification",
#     "dataset_info": get_dataset_info("cifar10"),
#     "data_module": datamodule,
#     "num_batchs": batch_size,
#     "epochs": 200,
# })



