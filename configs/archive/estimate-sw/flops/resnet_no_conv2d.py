import torch
from machop.models.vision.resnet.resnet import BasicBlock, ResNet
from torch.nn import Sequential

ignore_modules = [ResNet, Sequential, BasicBlock, torch.nn.Conv2d]


config = dict(
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=1,
    warm_up=10,
    as_string=True,
    output_file="estimate-sw_reports/resnet_no_conv2d.txt",
    ignore_modules=ignore_modules,
)
