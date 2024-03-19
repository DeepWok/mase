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

ONNXRT offers  extensive hardware support, accommodating not just NVIDIA GPUs but also CPUs, AMD GPUs, and other accelerators, thus it is an ideal choice if you're working without a NVIDIA GPU. Additionally, ONNXRT has high interoperability, allowing for model conversions through the ONNX framework, such as from PyTorch to TensorFlow. However, NVIDIA has significantly optimized TensorRT specifically for their devices, achieving higher throughput and lower latency compared to ONNXRT.

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

The supported modules can be converted to FP32, fp16, or int8.

Mixed precision is also supported, both for layerwise (by name) and typewise (by type). This means that you could, for example, quantize a CNN by setting all convolutional layers to fp16, and set the linear layers to int8 (typewise) or by setting all but the last two layers to int8. 

## ⚙️ How It Works

### TensorRT
<div align="center">
    <img src='../imgs/tensorrt_flow.png' width='300'>
</div>


**Fake Quantization**

To minimise losses during quantization, we first utilise Nvidia's Pytorch-Quantization framework to convert the model to a fake-quantized form.Fake quantization is used to perform calibration and fine tuning (QAT) before actually quantizing. The [Pytorch-Quantization](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#) libray simply emulates and prepares for quantization - which can then later be converted to ONNX and passed through to TensorRT. 

*Note:* This is only used if we have int8 quantized modules, as other precisions are not currently supported within the library.

This is acheived through the `tensorrt_fake_quantize_transform_pass` which goes through the model, either by type or by name, replaces each layer appropriately to a fake quantized form if the `quantize` parameter is set in the default config (`passes.tensorrt.default.config`) or on a per name or type basis. 

**Calibration**
Calibration is the TensorRT terminology of passing data samples to the quantizer and deciding the best amax for activations.

Calibrators can be added as a search space parameter to examine the best performing calibrator. These include `max`, `entropy`, `percentile` and `mse`.

**Quantization-aware Training**

Quantization-aware training (QAT) achieves the highest accuracy compared to dynamic quantization (whereby the quantization occurs just before doing compute), and Post Training Static Quantization or (PTQ) (whereby the model is statically calibrated).

In QAT, during both forward and backward training passes, weights and activations undergo "fake quantization" (although they are rounded to simulate int8 values, computations continue to utilize floating point numbers). Consequently, adjustments to the weights throughout the training process take into account the eventual quantization of the model. As a result, this method often leads to higher accuracy post-quantization compared to the other two techniques.

Since float quantization does not require calibration, nor is it supported by `pytorch-quantization`, models that do not contain int8 modules will not undergo fake quantization, unfortunately, for the time being this means QAT is unavailable and only udergoes Post Training Quantization (PTQ).

The `tensorrt_fine_tune_transform_pass` is used to fine tune the quantized model. 

For QAT it is typical to employ 10% of the original training epochs, starting at 1% of the initial training learning rate, and a cosine annealing learning rate schedule that follows the decreasing half of a cosine period, down to 1% of the initial fine tuning learning rate (0.01% of the initial training learning rate). However this default can be overidden by setting the `epochs`, `initial_learning_rate` and `final_learning_rate` in `passes.tensorrt.fine_tune`.

The fine tuned checkpoints are stored in the ckpts/fine_tuning folder:

```
mase_output
└── tensorrt
    └── quantization
        ├── cache
        ├── ckpts
        │   └── fine_tuning
        ├── json
        ├── onnx
        └── trt
```

**TensorRT Quantization**

After QAT, we are now ready to convert the model to a tensorRT engine so that it can be run with the superior inference speeds. To do so, we use the `tensorrt_engine_interface_pass` which converts the `MaseGraph`'s model from a Pytorch one to an ONNX format as an intermediate stage of the conversion.

During the conversion process, the `.onnx` and `.trt` files are stored to their respective folders shown above. This means that the `.onnx` files can be utilised for other model types and does not need to be just an unutilized, intermediary step.

This interface pass returns a dictionary containing the `onnx_path` and `trt_engine_path`.

**Performance Anaylisis**
To showcase the improved inference speeds and to evaluate accuracy and other performance metrics, the `runtime_analysis_pass` can be used. The pass can take a MaseGraph as an input, as well as an ONNX graph. For this comparison, we will first run the anaylsis pass on the original unquantized model and then on the int8 quantized model.


## 🚀 Getting Started
The environment setup during the [MASE installation](../../README.md) either through Docker or Conda will have you covered, there are no other requirements. 

The procedure in the [How It Works Section](#⚙️-how-it-works) can be acomplished by the machop's transform action.

```python
./ch transform --config {config_file} --load {model_checkpoint} --load-type pl
```

### Tutorials
We strongly recommend you look through the dedicated tutorials which walk you through the process of utilising MaseRT:
- [TensorRT Tutorial](/docs/tutorials/tensorrt/tensorRT_quantization_tutorial.ipynb) 
- [ONNXRT Tutorial](/docs/tutorials/)