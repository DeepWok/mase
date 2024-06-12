# Group5 README

Team Member:

- Xinyi Sun(xs1423@ic.ac.uk)
- Konstantinos Rotas(kr915@ic.ac.uk)

# General introduction

This study presents a new method to make machine learning models run faster on NVIDIA graphics cards. We've designed a system that speeds up the process of running and improving these models using a tool called TensorRT, without the need to retrain them. It works well for both simple and complex models, and at different levels of detail. We also looked into advanced techniques like adjusting precision on a layer-by-layer basis and integrating with ONNXRuntime to boost performance even more. This method is an important advancement for using machine learning in real-time on NVIDIA GPU.

Detailed descriptions of the calibration and quantization processes are available as HTML files stored in mase/machop/Group5_Documents_html.

# Preparation and installations

- **Python**: Python 3.10.14
- **GPU Specifications**: NVIDIA GeForce RTX 3050 Ti Laptop GPU, supported by CUDA version 12.1
- **TensorRT**: Version 8.6.1 for optimizing deep learning models for inference.
- **ONNX**: Version 1.15.0, for model exporting to a format compatible with various inference engines.
- **PyTorch**: Version 2.2.1, the primary deep learning framework used for model development and training.
- **Pytorch-Quantization**: Version 2.1.3, a toolkit for implementing quantization within PyTorch models to reduce model size and accelerate inference.
- **CUDA-Python**: Version 12.4.0, providing Python bindings to CUDA functionalities, further enabling efficient GPU-accelerated computing.
- **Additional Dependencies**: All other packages were installed according to the requirements specified by the MASE graph framework.

# PASS for Mase Graph

**TensorRT Quantize Pass:**
This step involves analyzing the provided input configuration to select the layers targeted for quantization. It identifies the precision (quantization bits) for both inputs and weights and decides if quantization should proceed. Subsequently, the targeted layer in the MASE graph is replaced with a `quant_nn` layer, tailored for quantized operations.

**Calibration Pass:**
Utilizing the specified data module and batch size, this process determines the distribution range of tensor values across each channel. It computes an `Amax` parameter that defines the quantization bounds, ensuring that the quantized model accurately reflects the original data distribution.

**Export to ONNX Pass:**
This step adjusts the input tensor dimensions according to a sample input derived from the training dataset and exports the model in the ONNX format to a predetermined location. This file path is registered within the MASE graph's metadata, ensuring the model's ONNX representation is available for future use.

**Generate TensorRT String Pass:**
This step serializes the TensorRT engine and stores this representation within the MASE graph's metadata. This serialization format enables automated requests of the model engine in future tasks.

**Run TensorRT Pass:**
Initiating with the MASE graph as input, this function deserializes the stored engine string and sets the necessary data buffers for intermediate data storage during inference. The engine then executes model predictions using a test dataset, to record accuracy and latency metrics.

## RUN FILES

- terminal path: ..\mase\machop
- Files for testing functions:
  - ..\mase\machop\Test_bynames.py (VGG)
  - ..\mase\machop\Test_byTypes.py (VGG)
  - ..\mase\machop\Test_JSC_ByType (JSC)

## Documents

**Related Path: mase\machop\Group5_Documents_html**

In order to pass the pull request, some of the associated JAVA generated files and related programs have been removed, but the HTML files themselves can still be read normally.
