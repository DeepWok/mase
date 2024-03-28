# TensorRT Integration to MASE

## Introduction

In this project, we integrates TensorRT into MASE to enable faster inference. We have implemented several passes that can be used to transform a PyTorch model to a TensorRT engine. 

## Getting Started

To get started, the following packages need to be installed additional to the MASE requirements:
- `pytorch-quantization==2.2.0`
- `pycuda==2024.1`

### Implemented Passes

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


mg = mixed_precision_transform_pass(mg, pass_args_mixed_precision, pass_args_calibrate)
```


