# TensorRT Integration to MASE

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


