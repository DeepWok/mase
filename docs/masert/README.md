# MaseRT: TensorRT and ONNXRT

## Overview

<div style="width: 100%; table-layout: fixed; margin: 0; padding: 0; border-collapse: collapse; display: table;">
  <div style="display: table-row; margin: 0; padding: 0;">
    <div style="display: table-cell; vertical-align: top; width: 70%; margin: 0; padding: 0;">
      <p>This documentation details the rationality, functionality and methodology of TensorRT and ONNXRT integration into the Machop framework.</p>
      <p>Continuing the ideology behind MASE to provide a reliable and efficient streaming accelerator system, we have now integrated TensorRT and ONNXRT, two powerful SDKs for optimizing inference using techniques such as quantization, layer and tensor fusion, and kernel tuning.</p>
      <p>The integration of both frameworks has shown to produce 2-4x inference speeds with higher energy efficiency whilst not significantly compromising model accuracy.</p>
    </div>
    <div style="display: table-cell; vertical-align: top; width: 30%; margin: 0; padding: 0;">
      <img src='../imgs/mase_rt_logo.png' style="height: auto; max-width: 100%; max-height: 100%;">
    </div>
  </div>
</div>


## Why Should I Care About Runtime Frameworks?

Runtime frameworks such as ONNX (Open Neural Network Exchange) Runtime and Nvidia's TensorRT are essential for streamlining deep learning model deployment, offering key benefits:

📈 **Quantization**: Quantization is the process of converting a model 
. Both runtime frameworks support model quantization, improving speed and reduce size without major accuracy losses.

🚀 **Speed**: They also accelerate model inference via optimization techniques like layer fusion and kernel auto-tuning, enhancing response times and throughput without the need for in-depth knowledge of CUDA.

💾 **Efficiency**: Lowers memory through model size reduction during quantization and further optimizes memory and computational resources, enabling deployment on devices with limited capacity.

🔧 **Interoperability**: With ONNX support, the Pytorch MaseGraph model is able to be converted to other frameworks such as TensorFlow, Optimum and Keras, and [many others](https://onnx.ai/supported-tools.html).

## 🛠️ Key functionality

### 💻 Hardware

| Device Type    | ONNXRT | TensorRT      |
|----------------|-----------------------|---------------|
| CPU            | ✅                     | ❌             |
| GPU (Generic)  | ✅                     | ❌             |
| NVIDIA GPU     | ✅                     | ✅             |

##  <img src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/related_work/vendor_specific_apis/tensorrt.png" width="20" height="20"> TensorRT

**Module Support** 
| Layer       | Modules                      |
|--------------|-------------------------------|
| Linear       | Linear                        |
| Convolution  | Conv1d, Conv2d, Conv3d        |
| Transpose Convolution | ConvTranspose1d, ConvTranspose2d, ConvTranspose3d |
| Pooling (Max) | MaxPool1d, MaxPool2d, MaxPool3d |
| Pooling (Average) | AvgPool1d, AvgPool2d, AvgPool3d |
| LSTM         | LSTM, LSTMCell                |

Currently, Pytorch-Quantization only supports the modules above, however custom quantized module can be made, find out more [here](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#document-tutorials/creating_custom_quantized_modules).

**Precision** 

The supported modules can be converted to FP32, FP16, or INT8.

Mixed precision is also supported, both for layerwise (by name) and typewise (by type). This means that you could, for example, quantize a CNN by setting all convolutional layers to FP16, and set the linear layers to INT8 (typewise) or by setting all but the last two layers to INT8. 

## ⚙️ How It Works

### TensorRT
**Quantization-aware Training**

Quantization-aware training (QAT) achieves the highest accuracy compared to dynamic quantization (whereby the quantization occurs just before doing compute), and Post Training Static Quantization or (PTQ) (whereby the model is statically calibrated).

In QAT, during both forward and backward training passes, weights and activations undergo "fake quantization": although they are rounded to simulate int8 values, computations continue to utilize floating point numbers. Consequently, adjustments to the weights throughout the training process take into account the eventual quantization of the model. As a result, this method often leads to higher accuracy post-quantization compared to the other two techniques.

<img src='../imgs/tensorrt_flow_chart.png' width='200'>

Fake quantization is used to perform calibration and fine tuning (QAT) before actually quantizing. The [Pytorch-Quantization](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#) libray simply emulates and prepares for quantization - which can then later be converted to ONNX and passed through to TensorRT. This is only used if we have INT8 calibration, as other precisions are not currently supported within the library.

This is acheived through the `tensorrt_fake_quantize_transform_pass` which goes through the model, either by type or by name, replaces each layer appropriately to a fake quantized form if the `quantize` parameter is set in the default config (`passes.tensorrt_quantize.default.config`) or on a per name or type basis. 

## 🚀 Getting Started

The environment setup during the [MASE installation](../../../README.md) either through Docker or Conda will have you covered, there are no other requirements. 

### Tutorials
We strongly recommend you look through the dedicated tutorials which walk you through the process of utilising MaseRT:
- [TensorRT tutorial](/docs/tutorials/tensorrt/tensorRT_quantization_tutorial.ipynb) 

### Which Runtime Should I Use

- Hardware: ONNXRT supports a wider range of hardware beyond NVIDIA GPUs, including CPUs, AMD GPUs, and other accelerators. If you do not have a NVIDIA GPU, use ONNXRT.
- Cross-Frameworks: ONNXRT allows conversion to other models through their ONNX framework (i.e., from PyTorch to Tensorflow)
- Cross-Platform Deployment: ONNX enables easier deployment across different operating systems and environments.
- Sheer Power: NVIDIA has heavily optimised TensorRT for their devices so their runtime has shown to acheive higher throughput, and lower latency copmared with ONNXRT.