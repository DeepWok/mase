import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
import time
import onnx
import tensorrt as trt
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
from chop.passes.graph.transforms.quantize.quantize_tensorRT import export_to_onnx_pass,generate_tensorrt_string_pass,run_tensorrt_pass
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
"conv2d":{
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
},}

}

###########################################################
#              Define a Search Strategy                   #
###########################################################

def run_model(mg, device, data_module, num_batches):
    j = 0
    accs, losses = [], []
    mg.model = mg.model.to(device)
    mg.model.eval()  # Set the model to evaluation mode
    inputs_tuple = ()
    for inputs in data_module.test_dataloader():
        xs, ys = inputs
        xs, ys = xs.to(device), ys.to(device)
        preds = mg.model(xs)
        loss = torch.nn.functional.cross_entropy(preds, ys)
        _, predicted = torch.max(preds, 1)  # Get the predicted classes
        correct = (predicted == ys).sum().item()  # Compute the number of correct predictions
        total = ys.size(0)  # Total number of images
        acc = correct / total  # Compute the accuracy
        accs.append(acc)  # Use .item() to get a Python number from a tensor
        losses.append(loss.item())  # Use .item() to get a Python number from a tensor
        # if j >= num_batches:  # Use >= instead of > to ensure num_batches iterations are done
        #     break
        j += 1
        inputs_tuple += (inputs,)
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    return acc_avg, loss_avg, inputs_tuple



import torch
from torchmetrics.classification import MulticlassAccuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)
import pdb; pdb.set_trace()
metric = MulticlassAccuracy(num_classes=5)
metric = metric.to(device)
num_batchs = 5
accuracy_list = []
latency_list = []

#########################################################
#      Experiment PART 1: Original Graph
#########################################################

##
# Ori Network
##


# engine_str_path_ori = './testdemo_engine_ori.trt'
# mg, _ = export_to_onnx_pass(mg, dummy_in, input_generator, onnx_model_path = './testdemo.onnx')
# mg,_ = generate_tensorrt_string_pass(mg, TR_output_path = './testdemo_engine.trt')
# run_tensorrt_pass(mg, dataloader = data_module.test_dataloader())

###
# Linear Only: Quantize to 8 bites
###

pass_args = {
"by": "type",
"default": {"config": {"name": None}},
# "linear": {
#         "config": {
#             "name": "integer",
#             # data
#             "data_in_width": 6,
#             "data_in_frac_width": 4,
#             # weight
#             "weight_width": 6,
#             "weight_frac_width": 4,
#             # bias
#             "bias_width": 6,
#             "bias_frac_width": 4,
#         },
#         "fake": False
# },
"conv2d":{
        "config": {
            "name": "integer",
            # data
            "data_in_width": 6,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 6,
            "weight_frac_width": 4,
            # bias
            "bias_width": 6,
            "bias_frac_width": 4,
},
"fake": False}

}
widths = [14, 12, 10, 8, 6, 4, 2]
acc = []
for width in widths:
    # pass_args["linear"]["config"]["data_in_width"] = width
    # pass_args["linear"]["config"]["weight_width"] = width
    # pass_args["linear"]["config"]["bias_width"] = width

    pass_args["conv2d"]["config"]["data_in_width"] = width
    pass_args["conv2d"]["config"]["weight_width"] = width
    pass_args["conv2d"]["config"]["bias_width"] = width
    mg, _ = tensorRT_quantize_pass(mg, pass_args,fake = True)
    mg, _ = calibration_pass(mg, pass_args,data_module,batch_size)
    acc_avg, loss_avg, inputs_tuple = run_model(mg,device,data_module,num_batchs)
    acc.append(acc_avg)
    # print("test_acc:",acc_avg)
    # engine_str_path = './testdemo_engine.trt'
    # mg, _ = export_to_onnx_pass(mg, dummy_in, input_generator, onnx_model_path = './testdemo.onnx')
    # mg,_ = generate_tensorrt_string_pass(mg, TR_output_path = './testdemo_engine.trt')
    # acc,latency = run_tensorrt_pass(mg, dataloader = data_module.test_dataloader())   
    # accuracy_list.append(acc)
    # latency_list.append(latency)

print("-------------------------------------------------------------")
print(widths)
print(accuracy_list)
print(latency_list)

print(acc_avg)