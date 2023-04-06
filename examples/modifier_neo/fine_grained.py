import os
import sys
from pprint import pprint as pp

import colorlog
import toml
import torch

sys.path.append("../../software")
import torch.fx as fx
from machop.models.vision.resnet import get_resnet18
from machop.modify.modifier import Modifier
from pytorch_lightning import seed_everything

seed_everything(0)

config_path = "./fine_grained.toml"

config = toml.load(config_path)
print("=" * 40)
print("How does a fine grained toml look?")
pp(config)

resnet18 = get_resnet18({"num_classes": 10})
modifier = Modifier(model=resnet18, config_path=config_path)
graph_module = modifier.modify()

print("=" * 40)
print("How does this modified model look?")
pp(dict(graph_module.named_children()))
