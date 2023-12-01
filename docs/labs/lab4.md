<!-- # Lab 1 for Advanced Deep Learning Systems (ADLS ELEC70109/EE9-AML3-10/EE9-AO25) -->

<br />
<div align="center">
  <a href="https://deepwok.github.io/">
    <img src="../imgs/deepwok.png" alt="Logo" width="160" height="160">
  </a>

  <h1 align="center">Lab 4 for Advanced Deep Learning Systems (ADLS) - Software Stream</h1>

  <p align="center">
    ELEC70109/EE9-AML3-10/EE9-AO25
    <br />
		Written by
    <a href="https://aaron-zhao123.github.io/">Aaron Zhao </a>
  </p>
</div>

# General introduction

In this lab, you will learn how to use the search functionality in the software stack of MASE to implement a Network Architecture Search.

There are in total 4 tasks you would need to finish, there is also 1 optional task.


# What is Network Architecture Search?

The design of a network architecture can greatly impact the performance of the model.
Consider the following optimization problem:

$\min_{a \in \mathcal{A}} \mathcal{L}_{val}(w^*(a), a)$

$s.t.~w^*(a) = \argmin_{w}(\mathcal{L}_{train}(w, a))$

For an architecture $a$ sampled from a set of architectures $\mathcal{A}$, we are minimizing the loss value when the architecture $a$ is parameterized by $w^*(a)$.

Meanwhile, the parameters $w^*(a)$ is the particular parameterization that provides the lowest $\mathcal{L}_{train}$ given the architecture $a$.

This is the core optimization problem involved in Network Architecture Search, where several approximations can happen. For instance, we can approximate $\mathcal{L}_{train}$ or $\mathcal{L}_{val}$, or the $\min_{a \in \mathcal{A}}$ can be formulated as a reinforcement learning process and so on.

## A Handwritten JSC Network 

We follow a similar procedure of what you have tried in lab3 to setup the dataset, copy and paste the following code snippet to a file, and name it `lab4.py`.

```python
import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

# figure out the correct path
machop_path = Path(".").resolve().parent.parent /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger

from chop.passes.transforms.interface import save_node_meta_param_transform_pass
from chop.passes.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.passes.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model

set_logging_verbosity("info")

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}
```

This time we are going to use a slightly different network, so we define it as a Pytorch model, copy and paste this snippet also to `lab4.py`.

> [Note]
MASE integrates seamlessly with native Pytorch models.

```python

from torch import nn
from chop.passes.utils import get_parent_name

# define a new model 
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear  2
            nn.Linear(16, 16),  # linear  3
            nn.Linear(16, 5),   # linear  4
            nn.ReLU(5),  # 5
        )

    def forward(self, x):
        return self.seq_blocks(x)


model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg = init_metadata_analysis_pass(mg, None)
```

# Model Architecture Modification as a Transformation Pass

Similar to what you have done in `lab2`, one can also implement a change in model architecture as a transformation pass:

```python

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features, 
        out_features=out_features, 
        bias=bias)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"] 
            elif name == "both":
                in_features = in_features * config["channel_multiplier"] 
                out_features = out_features * config["channel_multiplier"] 
            elif name == "input_only":
                in_features = in_features * config["channel_multiplier"] 
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph



pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        }
    },
"seq_blocks_3": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}

# this performs the architecture transformation based on the config
mg = redefine_linear_transform_pass(
    graph=mg, pass_args={"config": pass_config})
```

Copy and paste the above coding snippet and run your code.
The modified network features linear layers expanded to double their size, yet it's unusual to sequence three linear layers consecutively without interposing any non-linear activations (do you know why?).

So we are interested in a modified network:

```python
# define a new model 
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )

    def forward(self, x):
        return self.seq_blocks(x)
```

1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the `ReLU` also.

2. In `lab3`, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?

3. You may have noticed, one problem with the channel multiplier is that it scales all layers uniformly, ideally, we would like to be able to construct networks like the following:
    ```python
    # define a new model
    class JSC_Three_Linear_Layers(nn.Module):
        def __init__(self):
            super(JSC_Three_Linear_Layers, self).__init__()
            self.seq_blocks = nn.Sequential(
                nn.BatchNorm1d(16),
                nn.ReLU(16), 
                nn.Linear(16, 32),  # output scaled by 2
                nn.ReLU(32),  # scaled by 2
                nn.Linear(32, 64),  # input scaled by 2 but output scaled by 4
                nn.ReLU(64),  # scaled by 4
                nn.Linear(64, 5),  # scaled by 4
                nn.ReLU(5),  
            )

        def forward(self, x):
            return self.seq_blocks(x)
    ```
    Can you then design a search so that it can reach a network that can have this kind of structure?

4. Integrate the search to the `chop` flow, so we can run it from the command line.


# Optional task (scaling the search to real networks)

We have looked at how to search, on the architecture level, for a simple linear layer based network. MASE has the following components that you can have a look:

- [Cifar10 dataset](../../machop/chop/dataset/vision/cifar.py)
- [VGG](../../machop/chop/models/vision/vgg_cifar/vgg_cifar.py), this is a variant used for CIFAR
- [TPE-based Search](../../machop/chop/actions/search/strategies/optuna.py), implementd using [Optuna](https://optuna.readthedocs.io/en/stable/reference/index.html)

Can you define a search space (maybe channel dimension) for the VGG network, and use the TPE-search to tune it?
