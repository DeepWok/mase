# TensorRT Integration to MASE

## Introduction

In this project, we integrates TensorRT into MASE to enable faster inference. We have implemented several passes that can be used to transform a PyTorch model to a TensorRT engine. 

## Getting Started

To get started, the following packages need to be installed additional to the MASE requirements:
- `pytorch-quantization==2.2.0`
- `pycuda==2024.1`

## Main Achievements
All basic requirements for the project.

- Implement a transformation pass that achieves whole model quantization along with calibration.
- An interface pass to export the mase graph to a TensorRT Engine. 
- Implemented a pass that estimate the performance of TensorRT engine,

For extention, we implement layer-wise mixed-precision transform.


## Implemented Passes

- `mixed_precision_transform_pass`: A pass that transforms the graph to mixed precision. Based on fx_graph and pytorch-quantization package.
- `quantization_aware_training_pass`: A pass that performs quantization aware training. Based on pytorch-quantization package.
- `graph_to_trt_pass`: A pass that generates a TensorRT engine from a fx graph.
- `evaluate_pytorch_model_pass`: A pass that evaluates the PyTorch model on a given dataset. Can be used to test the accuracy of fake quantized models.
- `test_trt_engine`: A function that tests a TensorRT engine on a given dataset.
- `quantize_tensorrt_transform_pass`: A pass that quantizes a given pytorch model and optimizes by TensorRT automatically.

### Usage

#### 1. Controllable Mixed Precision Transform

To enable mixed precision conversion, we can use the `mixed_precision_transform_pass`. This pass transforms the graph to mixed precision, and require two additional arguments: `mixed precision options` and `calibration options`. Both are in dictionary format. The `mixed precision options` specifies the precision of target layers and support `fp32`, `fp16` and `int8`.

A simple usage example is as follows, and the following code finishes the mixed precision transformation within pytorch format.

```python
pass_args_mixed_precision = {
    "by": "name",
    "default": {"config": {"name": None}},
    "feature_layers_0": {
        "config": {
            "FakeQuantize": True,
            "name": "int",
            "input": {
                "precesion": 8,
                "calibrator": "max",
                "quantize_axis": None,
            },
            "weight": {
                "calibrator": "max",
                "quantize_axis": None,
            },
        }
    }
}

pass_args_calibrate = {
    "calibrator": "",
    "percentiles": [99],
    "data_module": data_module,
    "num_batches": 100,
}


mg, _ = mixed_precision_transform_pass(mg, pass_args_mixed_precision, pass_args_calibrate)
```
#### 2. Controllable Mixed Precision Conversion to TensorRT Engine

To enable TensorRT engine conversion, we can use the `graph_to_trt_pass`. This pass generates a TensorRT engine from a fx graph, then to ONNX model and TensorRT built-in ONNX parser is utilized to transform ONNX model to engine. 

To utilize the pass, we only need to provide the `engine_path` and `ONNX_path`. The `engine_path` specifies the path to store the TensorRT engine, and the `ONNX_path` specifies the path to store the ONNX model. Example usage is as follows:

```python

pass_args = {
    "onnxFile": "onnx_a_3_1.onnx",
    "engineFile": "engine_a_3_1.plan",
}
mg, _ = graph_to_trt_pass(mg, pass_args)
```

Estimating the performance of TensorRT engine is supported by the `test_trt_engine` function. This function takes the TensorRT engine and a dataset as input, and returns the performance metrics of the engine including the accuracy an latency. To address the issue of GPU and CPU desynchronization when running TensorRT engines in a Python environment, we utilize the PyCUDA library to calculate the GPU runtime, thereby obtaining the accurate runtime of the TensorRT engine. To use this function, we need to provide the `engine_path` and `dataset` as input. Example usage is as follows:

```python
test_trt_engine("engine_a_3_1.plan", data_module.test_dataloader)
```

#### 3. TensorRT command-line tool for detailed profiling

To enable TensorRT engine profiling, we can use the `trtexec` command-line tool. This tool is provided by the TensorRT package and can be used to profile the layer-weise performance of TensorRT engines. To install this tool, refer to the official TensorRT documentation [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tensorrt-python-api).

Two examples of profiling TensorRT engines are as follows:

- Profiling a TensorRT engine from ONNX model:

```
trtexec --onnx=/home/qizhu/Desktop/Work/mase/docs/adl_project/onnx_test.onnx --saveEngine=/home/qizhu/Desktop/Work/mase/docs/adl_project/engine_test.plan --profile --verbose
```

- Profiling a TensorRT engine from TensorRT engine:

```
trtexec --batch=8 --loadEngine=/home/qizhu/Desktop/Work/mase/docs/adl_project/engine_test.plan --exportProfile=/home/qizhu/Desktop/Work/mase/docs/adl_project/test_1.profile.json`
```

#### 4. Layer-wise Performance Estimation and Analysis

We visulized the layer-wise performance of TensorRT engines as well as the structure of engines. Two representitive results are shown below:

- Layer-wise performance of TensorRT engine:

![Layer-wise Performance of TensorRT Engine](https://raw.githubusercontent.com/liubingqi7/mase/f4a77aa6a82f0ee1a1c005fcaf252c6b843a409b/onnx_a_2_1.onnx.engine.graph.json.svg)

- Structure of TensorRT engine:

![Structure of TensorRT Engine](https://github.com/liubingqi7/mase/blob/proj/124.png?raw=true)