# Importing MASE as a Python package

An easy way to use MASE is to make a direct import.

As mentioned in Getting Started, you can install an editable version of MASE through pip by

```sh
pip install -e . -vvv
```

You can then test the installation by 

```sh
python -c"import chop; print(chop)"
```

## Transforming torch.Module

`chop.passes` offers a range of different passes that offer the capability to replace certain components for in the original neural network for various purposes. Some of these passes are Module Passes, that can directly operate on native `torch.nn.Module`, which basically means any arbitrary networks.

The following example is applying a MASE module pass to a pre-built `resnet50`. 

```python
import chop 

from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

from chop.passes import quantize_module_transform_pass

pass_args = {
    "by": "type",
    "linear": {
        "name": "integer",
        "data_in_width": 8,
        "data_in_frac_width": 4,
        "weight_width": 8,
        "weight_frac_width": 4,
        "bias_width": 8,
        "bias_frac_width": 4,
    },
    "conv2d": {
        "name": "integer",
        "data_in_width": 8,
        "data_in_frac_width": 4,
        "weight_width": 8,
        "weight_frac_width": 4,
        "bias_width": 8,
        "bias_frac_width": 4,
    },
}
transformed_model = quantize_module_transform_pass(model, pass_args)
print(transformed_model)
```

Transforming a MaseGraph

To support manipulation on a finer-level, eg. graph-level, MASE has provided built-in functionality to transformer any arbitrary `torch.nn.Module` into `MaseGraph`. This is because the ordinary `torch.Module` level view is normally not enough for finer manipulation -- many details are omitted at this level, we thus provide this `MaseGraph` to capture these detail. Correspondingly, we have provided a series of passes on the graph-level.


The following example is applying a MASE graph-level pass to a `vgg7` network. It also tries to use many MASE built-in functions to fetch the data, fetch the model, transform the model to `MaseGraph` land and then apply graph-level passes. 


```python
import logging

import chop 

from chop.dataset import MaseDataModule, get_dataset_info
from chop.ir.graph.mase_graph import MaseGraph
from chop import models
from chop.tools.get_input import InputGenerator, get_dummy_input

from chop.passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
    verify_common_metadata_analysis_pass,
)

from chop.passes.graph.utils import deepcopy_mase_graph

model_name = "vgg7"
dataset_name = "cifar10"
BATCH_SIZE = 32

# get dataset information
dataset_info = get_dataset_info(dataset_name)

# get model information
model_info = models.get_model_info(model_name)

# get data module
data_module = MaseDataModule(
    model_name=model_name,
    name=dataset_name,
    batch_size=BATCH_SIZE,
    num_workers=0,
    tokenizer=None,
    max_token_len=None,
)
data_module.prepare_data()
data_module.setup()
# NOTE: We only support vision classification models for now.
dummy_input = get_dummy_input(model_info, data_module, "cls", "cpu")

# get an input generator so that we can drive to get sample inputs
input_generator = InputGenerator(
    model_info=model_info,
    data_module=data_module,
        task="cls",
        which_dataloader="train",
    )

model = models.get_model(model_name, "cls", dataset_info, pretrained=True)

# This line transforms a nn.Module to a MaseGraph
mg = MaseGraph(model=model)

# Apply initialization passes to populate information in the graph
mg, _ = init_metadata_analysis_pass(mg, {})
mg, _ = add_common_metadata_analysis_pass(
    mg, {"dummy_in": dummy_input, "add_value": False}
)
# Sanity check and report
# mg = verify_common_metadata_analysis_pass(mg)
quan_args = {
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
    },
}

# deep copy is only possible if we put "add_value" to False
ori_mg = deepcopy_mase_graph(mg)
mg, _ = quantize_transform_pass(mg, quan_args)

summarize_quantization_analysis_pass(mg, pass_args={"save_dir": "quantize_summary", "original_mg": ori_mg})
```