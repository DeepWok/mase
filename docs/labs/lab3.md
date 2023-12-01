<!-- # Lab 1 for Advanced Deep Learning Systems (ADLS ELEC70109/EE9-AML3-10/EE9-AO25) -->

<br />
<div align="center">
  <a href="https://deepwok.github.io/">
    <img src="../imgs/deepwok.png" alt="Logo" width="160" height="160">
  </a>

  <h1 align="center">Lab 3 for Advanced Deep Learning Systems (ADLS)</h1>

  <p align="center">
    ELEC70109/EE9-AML3-10/EE9-AO25
    <br />
  Written by
    <a href="https://aaron-zhao123.github.io/">Aaron Zhao </a>
  </p>
</div>

# General introduction

In this lab, you will learn how to use the search functionality in the software stack of MASE.

There are in total 4 tasks you would need to finish.

# Writing a search using MaseGraph Transforms

In this section, our objective is to gain a comprehensive understanding of the construction of the current search function in Mase. To achieve this, we will require these essential components:

- MaseGraph: This component should be already created in the preceding lab.
- Search space: This component encompasses and defines the various available search options.
- Search strategy: An implementation of a search algorithm.
- Runner: This vital component manages and executes training, evaluation, or both procedures while generating a quality metric.

By analyzing these components, we can delve into the workings and effectiveness of the existing search function in Mase.

## Turning you network to a graph

We follow a similar procedure of what you have tried in lab2 to now produce a `MaseGraph`, this is converted from your pre-trained JSC model:

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
from chop.tools.logger import getLogger

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




logger = getLogger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint = None)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
```

You may want to copy the code snippet and paste it to a file created in the current directory with a name of `lab3.py`.

> [Warning: Directory madness]
The directory has to be correct because the line `machop_path = Path(".").resolve().parent.parent /"machop"` traces to the parent directory based on relative positions.

## Defining a search space

Based on the previous `pass_args` template, the following code is utilized to generate a search space. The search space is constructed by combining different weight and data configurations in precision setups.

```python
pass_args = {
"by": "type",
"default": {"config": {"name": None}},
"linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
},}

import copy
# build a search space
data_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
w_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
search_spaces = []
for d_config in data_in_frac_widths:
    for w_config in w_in_frac_widths:
        pass_args['linear']['config']['data_in_width'] = d_config[0]
        pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
        pass_args['linear']['config']['weight_width'] = w_config[0]
        pass_args['linear']['config']['weight_frac_width'] = w_config[1]
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_args))
```

## Defining a search strategy and a runner

The code provided below consists of two main `for` loops. The first `for` loop executes a straightforward brute-force search, enabling the iteration through the previously defined search space.

In contrast, the second `for` loop retrieves training samples from the train data loader. These samples are then utilized to generate accuracy and loss values, which serve as potential quality metrics for evaluating the system's performance.

```python
# grid search
ori_mg = deepcopy_mase_graph(mg)


import torch
from torchmetrics.classification import MulticlassAccuracy


metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5
# This first loop is basically our search strategy,
# in this case, it is a simple brute force search

recorded_accs = []
for i, config in enumerate(search_spaces):
    mg = quantize_transform_pass(ori_mg, config)
    j = 0

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds = mg.model(xs)
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batchs:
            break
        j += 1
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    recorded_accs.append(acc_avg)
```

Now if you copy also this code snippet into `lab3.py`, you would have a complete search scripts.

We now have the following task for you:

1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It's important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).

# The search command in the MASE flow

The search flow implemented in MASE is very similar to the one that you have constructed manually, the overall flow is implemented in [search.py](../../machop/chop/actions/search/search.py), the following bullet points provide you pointers to the code base.

- MaseGraph: this is the [MaseGraph](../../machop/chop/passes/graph/mase_graph.py) that you have used in lab2.
- Search space: The base class is implemented in [base.py](../../machop/chop/actions/search/search_space/base.py) , where in the same folder you can see a range of different supported search spaces.
- Search strategy: Similar to the search space, you can find a a base class [definition](../../machop/chop/actions/search/strategies/base.py), where different strategies are also defined in the same folder.
- Runner: Different [runners](../../machop/chop/actions/search/strategies/runners) can produce different metrics, they may also use `transforms` to help compute certain search metrics.

This enables one to execute the search through the MASE command line interface, remember to change the name after the `--load` option.

```bash
./ch search --config configs/examples/jsc_toy_by_type.toml --load your_pre_trained_ckpt
```

In this scenario, the search functionality is specified in the `toml` configuration file rather than via command-line inputs. This approach is adopted due to the multitude of configuration parameters that need to be set; encapsulating them within a single, elegant configuration file enhances reproducibility.

In `jsc_toy_by_type.toml`, the `search_space` configuration is set in `search.search_space`, the search strategy is configured via `search.strategy`. If you are not familiar with the `toml` syntax, you can read [here](https://toml.io/en/v1.0.0).

With now an understanding of how the MASE flow work, consider the following tasks

3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.
4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.
