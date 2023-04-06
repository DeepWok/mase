import sys
from pprint import pprint as pp

import toml
import torch

sys.path.append("../../software")
from machop.models.vision.resnet import get_resnet18
from machop.modify.modifier import Modifier
from pytorch_lightning import seed_everything

seed_everything(0)

config_path = "./coarse_grained.toml"

config = toml.load(config_path)
print("=" * 40)
print("How does a coarse grained toml look?")
pp(config)

resnet = get_resnet18({"num_classes": 10})

m = Modifier(
    resnet,
    config_path=config_path,
)
graph = m.modify()

print("=" * 40)
print("How does a model look after coarse-grained replacement?")
print(dict(graph.named_modules()))
# pp(template)
