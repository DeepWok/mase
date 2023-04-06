import sys

import torch

sys.path.append("../../software")
from machop.modify.modifier import Modifier
from pytorch_lightning import seed_everything

seed_everything(0)


# Replace CustomBlock with AnotherCustomBlock
class CustomBlock(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cb_1 = torch.nn.Linear(6, 8)
        self.cb_2 = torch.nn.Linear(6, 8)

    def forward(self, x):
        return self.cb_1(x) + self.cb_2(x)


class AnotherCustomBlock(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # set up layers using config
        self.ab_1 = torch.nn.Linear(6, 8)
        self.ab_2 = torch.nn.Linear(6, 8)
        print(config["name"] * 10)

    def forward(self, x):
        return self.ab_1(x) * self.ab_2(x)


# the model for generating graph
class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.Linear(4, 6)
        self.relu_1 = torch.nn.ReLU()
        self.cb = CustomBlock()

    def forward(self, x):
        x1 = self.l1(x)
        x_cb = self.cb(x1)
        y = torch.nn.functional.relu(x_cb)
        return y


# This function will be used by NeoModifier to generate layer for unknown layers
def create_custom_module(original_module, config):
    if type(original_module) is CustomBlock:
        return AnotherCustomBlock(config=config)
    else:
        raise RuntimeError


net = Net()
print("=" * 40)
print("The net model before modify-sw")
print(net)

# Create and save an empty template for this model
template = Modifier.create_empty_config_template(
    net,
    custom_leaf_module_classes=[CustomBlock, AnotherCustomBlock],
    save_path="./generated_template.toml",
)

m = Modifier(
    net,
    config_path="./replace_custom_block.toml",
    custom_leaf_module_classes=[CustomBlock, AnotherCustomBlock],
    create_new_custom_module_fn=create_custom_module,
)

graph_module = m.modify()
print("=" * 40)
print("The net model after block replacement")
print(dict(graph_module.named_modules()))
