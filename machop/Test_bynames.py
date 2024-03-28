import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
import time
import onnx
import tensorrt as trt
import json
from datetime import datetime
###########################################################
# Read me!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#
# The file is used for test the written pass for the final Lab
# suitable path is : mase/machop/USEDforTEST_lab3.py
############################################################


# figure out the correct path
# machop_path = Path(".").resolve().parent.parent /"machop"
machop_path = Path(".").resolve()
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model

from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)

from chop.passes.graph.transforms.quantize.quantize_tensorRT import tensorRT_quantize_pass,calibration_pass
from chop.passes.graph.transforms.quantize.quantize_tensorRT import export_to_onnx_pass,generate_tensorrt_string_pass,run_tensorrt_pass,run_model_for_test,run_tensorRT_without_String

from chop.tools.checkpoint_load import load_model

from chop.plt_wrapper import get_model_wrapper


logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 8 ### dont be 64 plz (maybe max is 32, nobody knows)
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

CHECKPOINT_PATH = "./VGG_checkPoint/test-accu-0.9332.ckpt"

model_info = get_model_info(model_name)

model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)

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


###########################################################
#                   Define a Search Space                   #
###########################################################
pass_args = {
"by": "name",
"default": {"config": {"name": None}},
"classifier_0": {
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

###########################################################
#              Define a Search Strategy                   #
###########################################################

import torch
from torchmetrics.classification import MulticlassAccuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

metric = MulticlassAccuracy(num_classes=5)
metric = metric.to(device)
num_batchs = 8
accuracy_tensorRT = []
latency_tensorRT = []
accuracy_runmodel = []
latency_runmodel = []
#########################################################
#      Experiment PART 1: Original Graph
#########################################################

###
# Linear Only: Quantize to 8 bites
###

acc_avg, loss_avg, latency_avg = run_model_for_test(mg, device, data_module, num_batchs)
accuracy_runmodel.append(acc_avg)
latency_runmodel.append(latency_avg)

pass_args = {
"by": "name",
"default": {"config": {"name": None}},
"classifier_0": {
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
        },
        "fake": "True",
        "calibration_method": "histogram",
},
# "classifier_2": {
#         "config": {
#             "name": "integer",
#             # data
#             "data_in_width": 8,
#             "data_in_frac_width": 4,
#             # weight
#             "weight_width": 8,
#             "weight_frac_width": 4,
#             # bias
#             "bias_width": 8,
#             "bias_frac_width": 4,
#         }
# },
}

widths = [8]
calibration = False
by = "name"
structure = "classifier_0_test"
fake = "fake_True"
calibration_str = "NOcalibration"

onnx_model_path = f'./OriginalMG.onnx'
trt_output_path = f'./OriginalMG.plan'

mg, _ = export_to_onnx_pass(mg, dummy_in, input_generator, onnx_model_path=onnx_model_path)
mg,_ = generate_tensorrt_string_pass(mg, TR_output_path=trt_output_path)
acc,latency = run_tensorrt_pass(mg, dataloader = data_module.test_dataloader()) 
accuracy_tensorRT.append(acc)
latency_tensorRT.append(latency)

for width in widths:
    pass_args["classifier_0"]["config"]["data_in_width"] = width
    pass_args["classifier_0"]["config"]["weight_width"] = width
    pass_args["classifier_0"]["config"]["bias_width"] = width

    # pass_args["classifier_2"]["config"]["data_in_width"] = width
    # pass_args["classifier_2"]["config"]["weight_width"] = width
    # pass_args["classifier_2"]["config"]["bias_width"] = width

    mg, _ = tensorRT_quantize_pass(mg, pass_args)
    if calibration == True:
        mg, _ = calibration_pass(mg,data_module,batch_size)  
    acc_avg, loss_avg, latency_avg = run_model_for_test(mg, device, data_module, num_batchs)
    
    # Generate unique file paths for each model
    # Define the directory paths
    onnx_dir_path = f'./ONNX_model/{by}_{fake}_{structure}_{calibration}/ONNX'
    trt_dir_path = f'./ONNX_model/{by}_{fake}_{structure}_{calibration}/Plan'

    # Check if the directories exist, if not, create them
    for dir_path in [onnx_dir_path, trt_dir_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Define the file paths
    onnx_model_path = f'{onnx_dir_path}/{by}_{fake}_{structure}_{width}_{calibration}.onnx'
    trt_output_path = f'{trt_dir_path}/{by}_{fake}_{structure}_{width}_{calibration}.plan'
    
    mg, _ = export_to_onnx_pass(mg, dummy_in, input_generator, onnx_model_path=onnx_model_path)
    mg,_ = generate_tensorrt_string_pass(mg, TR_output_path=trt_output_path)
    acc,latency = run_tensorrt_pass(mg, dataloader = data_module.test_dataloader())   
    accuracy_tensorRT.append(acc)
    latency_tensorRT.append(latency)
    accuracy_runmodel.append(acc_avg)
    latency_runmodel.append(latency_avg)

# # print experiment results
# print("widths: ",["Original Graph"] + widths)
# print("TensorRT Accuracy: ",accuracy_tensorRT)
# print("TensorRT Latency: ",latency_tensorRT)
# print("Run Model Accuracy: ",accuracy_runmodel)
# print("Run Model Latency: ",latency_runmodel)

# # Store results in a dictionary
# results = {
#     "widths": ["Original Graph"] + widths,
#     "TensorRT Accuracy": accuracy_tensorRT,
#     "TensorRT Latency": latency_tensorRT,
#     "Run Model Accuracy": accuracy_runmodel,
#     "Run Model Latency": latency_runmodel,
#     "pass_args": pass_args,  # Store pass_args information
#     "calibration": calibration,
#     "by": by,
#     "structure": structure,
#     "fake": fake,
#     "calibration_str": calibration_str
# }

# # Generate a timestamp
# timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# # Write results to a JSON file with a timestamp in its name
# with open(f'Pytorch_Quantization_Experiment_result/results_{pass_args["by"]}_{timestamp}.json', 'w') as f:
#     json.dump(results, f)