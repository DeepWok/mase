from nas_201_api import NASBench201API as API

from .......chop.dataset import MaseDataModule, get_dataset_info
from ......chop.tools.logger import set_logging_verbosity

from ......chop.passes.graph import (
    save_node_meta_param_interface_pass,
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from ......chop.tools.get_input import InputGenerator
from ......chop.tools.checkpoint_load import load_model
from ......chop.ir import MaseGraph

from ......chop.models import get_model_info, get_model

from ..zero_cost_nas.pruners.predictive import find_measures
from xautodl.models import get_cell_based_tiny_net

import torch.nn.functional as F

batch_size = 8
model_name = "vgg7"
dataset_name = "cifar10"

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)

data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

model = get_model(
    model_name,
    task="vision",
    dataset_info=data_module.dataset_info,
    pretrained=False)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="vision",
    which_dataloader="train",
)

# a demonstration of how to feed an input value to the model
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

api = API('/home/xz2723/mase_xinyi/machop/third_party/NAS-Bench-201-v1_1-096897.pth', verbose=False)
print("api loaded")
config = api.get_net_config(1, 'cifar10')
model_nasbench = get_cell_based_tiny_net(config)
dataload_info = ('random', 1, 10)
device = "cpu"
train_dataloader = model.train_dataloader()

value = find_measures(model_nasbench, 
                    train_dataloader,
                    dataload_info, # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
                    device, 
                    loss_fn=F.cross_entropy, 
                    measure_names='grad_norm',
                    measures_arr=None)