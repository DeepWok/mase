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
import torch
from torchmetrics.classification import MulticlassAccuracy

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
from chop.passes.graph.transforms.quantize.quantize_tensorRT import (
    tensorRT_quantize_pass,
    calibration_pass,
    export_to_onnx_pass,
    generate_tensorrt_string_pass,
    run_tensorrt_pass,
    run_model_for_test,
)
from chop.tools.checkpoint_load import load_model
from chop.plt_wrapper import get_model_wrapper

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
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint=None,
)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

mg = MaseGraph(model=model)

pass_args = {
    "by": "name",
    "default": {"config": {"name": None}},
    "classifier_0": {
        "config": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

acc_avg, loss_avg, latency_avg = run_model_for_test(mg, device, data_module, num_batchs)
accuracy_runmodel.append(acc_avg)
latency_runmodel.append(latency_avg)

pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 6,
            "data_in_frac_width": 4,
            "weight_width": 6,
            "weight_frac_width": 4,
            "bias_width": 6,
            "bias_frac_width": 4,
        },
        "fake": "True",
    },
}

widths = [10, 8, 6, 4, 2]
calibration = False
by = "type"
structure = "JSC_onlyLinear"
fake = "fake"

onnx_model_path = f"./OriginalMG.onnx"
trt_output_path = f"./OriginalMG.plan"

dir_path = f"./ONNX_model_JSCC/{by}_{fake}_{structure}_{calibration}/ONNX"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

dir_path = f"./ONNX_model_JSCC/{by}_{fake}_{structure}_{calibration}/Plan"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

acc_avg, loss_avg, latency_avg = run_model_for_test(mg, device, data_module, num_batchs)
mg, _ = export_to_onnx_pass(
    mg, dummy_in, input_generator, onnx_model_path=onnx_model_path
)
mg, _ = generate_tensorrt_string_pass(mg, TR_output_path=trt_output_path)
acc, latency = run_tensorrt_pass(mg, dataloader=data_module.test_dataloader())
accuracy_tensorRT.append(acc)
latency_tensorRT.append(latency)
accuracy_runmodel.append(acc_avg)
latency_runmodel.append(latency_avg)

for width in widths:
    pass_args["linear"]["config"]["data_in_width"] = width
    pass_args["linear"]["config"]["weight_width"] = width
    pass_args["linear"]["config"]["bias_width"] = width

    mg, _ = tensorRT_quantize_pass(mg, pass_args)
    if calibration == True:
        mg, _ = calibration_pass(mg, data_module, batch_size)
    acc_avg, loss_avg, latency_avg = run_model_for_test(
        mg, device, data_module, num_batchs
    )

    onnx_dir_path = f"./ONNX_model_JSCC/{by}_{fake}_{structure}_{calibration}/ONNX"
    trt_dir_path = f"./ONNX_model_JSCC/{by}_{fake}_{structure}_{calibration}/Plan"

    for dir_path in [onnx_dir_path, trt_dir_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    onnx_model_path = (
        f"{onnx_dir_path}/{by}_{fake}_{structure}_{width}_{calibration}.onnx"
    )
    trt_output_path = (
        f"{trt_dir_path}/{by}_{fake}_{structure}_{width}_{calibration}.plan"
    )

    mg, _ = export_to_onnx_pass(
        mg, dummy_in, input_generator, onnx_model_path=onnx_model_path
    )
    mg, _ = generate_tensorrt_string_pass(mg, TR_output_path=trt_output_path)
    acc, latency = run_tensorrt_pass(mg, dataloader=data_module.test_dataloader())
    accuracy_tensorRT.append(acc)
    latency_tensorRT.append(latency)
    accuracy_runmodel.append(acc_avg)
    latency_runmodel.append(latency_avg)

print("widths: ", ["Original Graph"] + widths)
print("TensorRT Accuracy: ", accuracy_tensorRT)
print("TensorRT Latency: ", latency_tensorRT)
print("Run Model Accuracy: ", accuracy_runmodel)
print("Run Model Latency: ", latency_runmodel)

results = {
    "widths": ["Original Graph"] + widths,
    "TensorRT Accuracy": accuracy_tensorRT,
    "TensorRT Latency": latency_tensorRT,
    "Run Model Accuracy": accuracy_runmodel,
    "Run Model Latency": latency_runmodel,
    "pass_args": pass_args,
    "calibration": calibration,
    "by": by,
    "structure": structure,
    "fake": fake,
}

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

with open(
    f'Pytorch_Quantization_Experiment_result/results_{pass_args["by"]}_{fake}_{calibration}_{structure}_{timestamp}.json',
    "w",
) as f:
    json.dump(results, f)
