# Welcome To the ONNX Runtime Tutorial!

This notebook is designed to demonstrate the features of the ONNXRT passes integrated into MASE as part of the MASERT framework.

## Section 1. ONNX Runtime Optimizations
Firstly, we will show you how we can utilise the ONNX RT optimizations. We expect to see a speed up without a loss in model accuracy. We will use a simple model, `jsc-toy`, and compare the optimized model to the original model using the `Machop API`.

First, we load the machop requirements by running the cell below.


```python
import sys
import os
from pathlib import Path
import toml

# Figure out the correct path
machop_path = Path(".").resolve().parent.parent.parent /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

# Add directory to the PATH so that chop can be called
new_path = "../../../machop"
full_path = os.path.abspath(new_path)
os.environ['PATH'] += os.pathsep + full_path

from chop.tools.utils import to_numpy_if_tensor
from chop.tools.logger import set_logging_verbosity
from chop.tools import get_cf_args, get_dummy_input
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph
from chop.models import get_model_info, get_model, get_tokenizer
from chop.dataset import MaseDataModule, get_dataset_info
from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    runtime_analysis_pass,
    )

set_logging_verbosity("info")
```

    /root/anaconda3/envs/mase/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


    [2024-03-28 00:02:22,727] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)


    [32mINFO    [0m [34mSet logging level to info[0m
    WARNING: Logging before flag parsing goes to stderr.
    I0328 00:02:24.891957 140309792896832 logger.py:44] Set logging level to info


We then load in a demonstration toml file and set the relevant pass arguments (this is all done automatically if we were to use the command line, see [Section 2](#section-2-int8-quantization))


```python
JSC_TOML_PATH = "../../../machop/configs/onnx/jsc_gpu_ort.toml"

# Reading TOML file and converting it into a Python dictionary
with open(JSC_TOML_PATH, 'r') as toml_file:
    pass_args = toml.load(toml_file)

# Extract the 'passes.tensorrt' section and its children
onnx_config = pass_args.get('passes', {}).get('onnxruntime', {})
# Extract the 'passes.runtime_analysis' section and its children
runtime_analysis_config = pass_args.get('passes', {}).get('runtime_analysis', {})

# Load the basics in
model_name = pass_args['model']
dataset_name = pass_args['dataset']
max_epochs = pass_args['max_epochs']
batch_size = pass_args['batch_size']
learning_rate = pass_args['learning_rate']
accelerator = pass_args['accelerator']

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)

data_module.prepare_data()
data_module.setup()

# Add the data_module and other necessary information to the configs
configs = [onnx_config, runtime_analysis_config]
for config in configs:
    config['task'] = pass_args['task']
    config['batch_size'] = pass_args['batch_size']
    config['model'] = pass_args['model']
    config['data_module'] = data_module
    config['dataset'] = pass_args['dataset']
    config['accelerator'] = 'cuda' if pass_args['accelerator'] == 'gpu' else pass_args['accelerator']
    if config['accelerator'] == 'gpu':
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
```

Next, we train the `jsc-toy` model using the machop `train` action with the config from the toml file. You may want to switch to GPU for this task - it will not affect the cpu optimizations later on.


```python
# !ch train --config {JSC_TOML_PATH} --accelerator gpu
```

Then we load in the checkpoint. You will have to adjust this according to where it has been stored in the mase_output directory.


```python
# Load in the trained checkpoint - change this accordingly
JSC_CHECKPOINT_PATH = "../../../mase_output/jsc-toy_cls_jsc/software/training_ckpts/best.ckpt"
model = load_model(load_name=JSC_CHECKPOINT_PATH, load_type="pl", model=model)

# Initiate metadata
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)
mg, _ = init_metadata_analysis_pass(mg, None)

# Copy original graph for analysis later
mg_original = deepcopy_mase_graph(mg)

mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)
mg, _ = metadata_value_type_cast_transform_pass(mg, pass_args={"fn": to_numpy_if_tensor})
```

    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from ../../../mase_output/jsc-toy_cls_jsc/software/training_ckpts/best.ckpt[0m
    I0327 14:20:09.068645 140012160939840 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from ../../../mase_output/jsc-toy_cls_jsc/software/training_ckpts/best.ckpt


We then run the `onnx_runtime_interface_pass` which completes the optimizations using the dataloader and `jsc-toy` model. This returns metadata containing the paths to the models:

- `onnx_path` (the optimized model)
- `onnx_dynamic_quantized_path` (the dynamically )

In this case, since we are not quantizing the model, only the `onnx_path` is available. 

The models are also stored in the directory:
```
mase_output
└── onnxrt
    └── model_task_dataset_date
        ├── optimized
        ├── pre_processed
        ├── static_quantized
        └── dynamic_quantized
```


```python
mg, onnx_meta = onnx_runtime_interface_pass(mg, pass_args=onnx_config)
```

    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0327 14:20:12.535338 140012160939840 onnx_runtime.py:48] Converting PyTorch model to ONNX...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/onnxrt/jsc-toy_cls_jsc_2024-03-27[0m
    I0327 14:20:12.539771 140012160939840 onnx_runtime.py:50] Project will be created at /root/mase/mase_output/onnxrt/jsc-toy_cls_jsc_2024-03-27
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/jsc-toy_cls_jsc_2024-03-27/optimized/version_1/model.onnx[0m
    I0327 14:20:12.751212 140012160939840 onnx_runtime.py:68] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/jsc-toy_cls_jsc_2024-03-27/optimized/version_1/model.onnx
    [32mINFO    [0m [34mONNX Model Summary: 
    +-------+----------------------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+---------------------+
    | Index |               Name               |        Type        |                                                          Inputs                                                          |                  Outputs                  |      Attributes     |
    +-------+----------------------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+---------------------+
    |   0   | /seq_blocks.0/BatchNormalization | BatchNormalization |            input, seq_blocks.0.weight, seq_blocks.0.bias, seq_blocks.0.running_mean, seq_blocks.0.running_var            | /seq_blocks.0/BatchNormalization_output_0 |  epsilon, momentum  |
    |   1   |        /seq_blocks.1/Relu        |        Relu        |                                        /seq_blocks.0/BatchNormalization_output_0                                         |        /seq_blocks.1/Relu_output_0        |                     |
    |   2   |        /seq_blocks.2/Gemm        |        Gemm        |                           /seq_blocks.1/Relu_output_0, seq_blocks.2.weight, seq_blocks.2.bias                            |        /seq_blocks.2/Gemm_output_0        | alpha, beta, transB |
    |   3   | /seq_blocks.3/BatchNormalization | BatchNormalization | /seq_blocks.2/Gemm_output_0, seq_blocks.3.weight, seq_blocks.3.bias, seq_blocks.3.running_mean, seq_blocks.3.running_var | /seq_blocks.3/BatchNormalization_output_0 |  epsilon, momentum  |
    |   4   |        /seq_blocks.4/Relu        |        Relu        |                                        /seq_blocks.3/BatchNormalization_output_0                                         |        /seq_blocks.4/Relu_output_0        |                     |
    |   5   |        /seq_blocks.5/Gemm        |        Gemm        |                           /seq_blocks.4/Relu_output_0, seq_blocks.5.weight, seq_blocks.5.bias                            |        /seq_blocks.5/Gemm_output_0        | alpha, beta, transB |
    |   6   | /seq_blocks.6/BatchNormalization | BatchNormalization | /seq_blocks.5/Gemm_output_0, seq_blocks.6.weight, seq_blocks.6.bias, seq_blocks.6.running_mean, seq_blocks.6.running_var | /seq_blocks.6/BatchNormalization_output_0 |  epsilon, momentum  |
    |   7   |        /seq_blocks.7/Relu        |        Relu        |                                        /seq_blocks.6/BatchNormalization_output_0                                         |        /seq_blocks.7/Relu_output_0        |                     |
    |   8   |        /seq_blocks.8/Gemm        |        Gemm        |                           /seq_blocks.7/Relu_output_0, seq_blocks.8.weight, seq_blocks.8.bias                            |        /seq_blocks.8/Gemm_output_0        | alpha, beta, transB |
    |   9   | /seq_blocks.9/BatchNormalization | BatchNormalization | /seq_blocks.8/Gemm_output_0, seq_blocks.9.weight, seq_blocks.9.bias, seq_blocks.9.running_mean, seq_blocks.9.running_var | /seq_blocks.9/BatchNormalization_output_0 |  epsilon, momentum  |
    |   10  |       /seq_blocks.10/Relu        |        Relu        |                                        /seq_blocks.9/BatchNormalization_output_0                                         |                     37                    |                     |
    +-------+----------------------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+---------------------+[0m
    I0327 14:20:12.757548 140012160939840 onnx_runtime.py:90] ONNX Model Summary: 
    +-------+----------------------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+---------------------+
    | Index |               Name               |        Type        |                                                          Inputs                                                          |                  Outputs                  |      Attributes     |
    +-------+----------------------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+---------------------+
    |   0   | /seq_blocks.0/BatchNormalization | BatchNormalization |            input, seq_blocks.0.weight, seq_blocks.0.bias, seq_blocks.0.running_mean, seq_blocks.0.running_var            | /seq_blocks.0/BatchNormalization_output_0 |  epsilon, momentum  |
    |   1   |        /seq_blocks.1/Relu        |        Relu        |                                        /seq_blocks.0/BatchNormalization_output_0                                         |        /seq_blocks.1/Relu_output_0        |                     |
    |   2   |        /seq_blocks.2/Gemm        |        Gemm        |                           /seq_blocks.1/Relu_output_0, seq_blocks.2.weight, seq_blocks.2.bias                            |        /seq_blocks.2/Gemm_output_0        | alpha, beta, transB |
    |   3   | /seq_blocks.3/BatchNormalization | BatchNormalization | /seq_blocks.2/Gemm_output_0, seq_blocks.3.weight, seq_blocks.3.bias, seq_blocks.3.running_mean, seq_blocks.3.running_var | /seq_blocks.3/BatchNormalization_output_0 |  epsilon, momentum  |
    |   4   |        /seq_blocks.4/Relu        |        Relu        |                                        /seq_blocks.3/BatchNormalization_output_0                                         |        /seq_blocks.4/Relu_output_0        |                     |
    |   5   |        /seq_blocks.5/Gemm        |        Gemm        |                           /seq_blocks.4/Relu_output_0, seq_blocks.5.weight, seq_blocks.5.bias                            |        /seq_blocks.5/Gemm_output_0        | alpha, beta, transB |
    |   6   | /seq_blocks.6/BatchNormalization | BatchNormalization | /seq_blocks.5/Gemm_output_0, seq_blocks.6.weight, seq_blocks.6.bias, seq_blocks.6.running_mean, seq_blocks.6.running_var | /seq_blocks.6/BatchNormalization_output_0 |  epsilon, momentum  |
    |   7   |        /seq_blocks.7/Relu        |        Relu        |                                        /seq_blocks.6/BatchNormalization_output_0                                         |        /seq_blocks.7/Relu_output_0        |                     |
    |   8   |        /seq_blocks.8/Gemm        |        Gemm        |                           /seq_blocks.7/Relu_output_0, seq_blocks.8.weight, seq_blocks.8.bias                            |        /seq_blocks.8/Gemm_output_0        | alpha, beta, transB |
    |   9   | /seq_blocks.9/BatchNormalization | BatchNormalization | /seq_blocks.8/Gemm_output_0, seq_blocks.9.weight, seq_blocks.9.bias, seq_blocks.9.running_mean, seq_blocks.9.running_var | /seq_blocks.9/BatchNormalization_output_0 |  epsilon, momentum  |
    |   10  |       /seq_blocks.10/Relu        |        Relu        |                                        /seq_blocks.9/BatchNormalization_output_0                                         |                     37                    |                     |
    +-------+----------------------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+---------------------+
    [33mWARNING [0m [34mQuantization is not set in default config. Skipping quantization.[0m
    W0327 14:20:12.758648 140012160939840 onnx_runtime.py:97] Quantization is not set in default config. Skipping quantization.


We can view a summary of the ONNX model (which is the unmodified from the Pytorch one), however it should be optimized. Let's run an analysis path on both the original `MaseGraph` and the `.onnx` optimized model.


```python
_, _ = runtime_analysis_pass(mg_original, pass_args=runtime_analysis_config)
```

    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy[0m
    I0327 14:20:16.984423 140012160939840 analysis.py:270] Starting transformation analysis on jsc-toy
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+---------------+
    |      Metric (Per Batch)      |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.73159    |
    |      Average Precision       |    0.74429    |
    |        Average Recall        |    0.73023    |
    |       Average F1 Score       |    0.73347    |
    |         Average Loss         |    0.76373    |
    |       Average Latency        |  0.79688 ms   |
    |   Average GPU Power Usage    |   21.816 W    |
    | Inference Energy Consumption | 0.0048292 mWh |
    +------------------------------+---------------+[0m
    I0327 14:20:19.793779 140012160939840 analysis.py:398] 
    Results jsc-toy:
    +------------------------------+---------------+
    |      Metric (Per Batch)      |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.73159    |
    |      Average Precision       |    0.74429    |
    |        Average Recall        |    0.73023    |
    |       Average F1 Score       |    0.73347    |
    |         Average Loss         |    0.76373    |
    |       Average Latency        |  0.79688 ms   |
    |   Average GPU Power Usage    |   21.816 W    |
    | Inference Energy Consumption | 0.0048292 mWh |
    +------------------------------+---------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-27/mase_graph/version_1/model.json[0m
    I0327 14:20:19.796502 140012160939840 analysis.py:84] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-27/mase_graph/version_1/model.json



```python
_, _ = runtime_analysis_pass(onnx_meta['onnx_path'], pass_args=runtime_analysis_config)
```

    /root/anaconda3/envs/mase/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:69: UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
      warnings.warn(
    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy-onnx[0m
    I0327 14:20:33.337222 140012160939840 analysis.py:270] Starting transformation analysis on jsc-toy-onnx


    [32mINFO    [0m [34m
    Results jsc-toy-onnx:
    +------------------------------+---------------+
    |      Metric (Per Batch)      |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.73412    |
    |      Average Precision       |    0.74875    |
    |        Average Recall        |    0.73435    |
    |       Average F1 Score       |    0.73761    |
    |         Average Loss         |    0.74954    |
    |       Average Latency        |   0.2215 ms   |
    |   Average GPU Power Usage    |   21.575 W    |
    | Inference Energy Consumption | 0.0013275 mWh |
    +------------------------------+---------------+[0m
    I0327 14:20:35.876071 140012160939840 analysis.py:398] 
    Results jsc-toy-onnx:
    +------------------------------+---------------+
    |      Metric (Per Batch)      |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.73412    |
    |      Average Precision       |    0.74875    |
    |        Average Recall        |    0.73435    |
    |       Average F1 Score       |    0.73761    |
    |         Average Loss         |    0.74954    |
    |       Average Latency        |   0.2215 ms   |
    |   Average GPU Power Usage    |   21.575 W    |
    | Inference Energy Consumption | 0.0013275 mWh |
    +------------------------------+---------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-27/onnx/version_0/model.json[0m
    I0327 14:20:35.878773 140012160939840 analysis.py:84] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-27/onnx/version_0/model.json


As shown above, the latency of the cpu inference is around 3.5x less with the `jsc-toy` model without compromising accuracy simply by using the optimizations of ONNXRT. 

Lets now run the same optimzations, this time using a GPU and a larger model - the `vgg7`.  We will also utilse the chop action from the terminal which runs the same `onnx_runtime_interface_pass` pass.

First lets train the `vgg7` model using the machop `train` action with the config from the new toml file and then load the trained checkpoint it into the `transform` pass.


```python
VGG_TOML_PATH = "../../../machop/configs/onnx/vgg7_gpu_quant.toml"

# !ch train --config {VGG_TOML_PATH}

# Load in the checkpoint from the previous train - modify accordingly
VGG_CHECKPOINT_PATH = "../../../mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt"

!ch transform --config {VGG_TOML_PATH} --load {VGG_CHECKPOINT_PATH} --load-type pl 
```

    /bin/bash: line 1: ch: command not found


As shown above, the latency of the gpu inference is 30% less with the `vgg7` model without compromising accuracy simply by using the optimizations of ONNXRT. 

We will now look at quantization to further speed up the model. 

## Section 2. Quantization

We may quantize either using FP16 or INT8 by setting the `precision` parameter in `passes.onnxruntime.default.config` to `'fp16'` or `'int8'` respectively. INT8 quantization will show the most notable latency improvements but is more likely to lower performance. 

There are three types of quantization for ONNXRT and can be set in `onnxruntime.default.config` under `quantization_types`. The differences of the first two are for how they calibrate i.e. set the scale and zero points which are only relevant for integer based quantization:
- **Static Quantization**:
    - The scale and zero point of activations are calculated in advance (offline) using a calibration data set.
    - The activations have the same scale and zero point during each forward pass.
    - The `num_calibration_batches` parameter must also be set to ensure calibration is tested on a subset of the training dataset. A larger subset will be beneficial for calibrating the amaxes and may improve accuracy, however it will result in a longer calibration time.
- **Dynamic Quantization**:
    - The scale and zero point of activations are calculated on-the-fly (online) and are specific for each forward pass.
    - This approach is more accurate but introduces extra computational overhead

The `onnx_runtime_interface_pass` pass also supports mixed precision. This is an automatic only procedure, where ONNXRT finds a minimal set of ops to skip while retaining a certain level of accuracy, converting most of the ops to float16 but leaving some in float32. 
- **Auto Mixed Precision Quantization**:
    - Automatically adjusts between FP16 and FP32 precisions to retain certain level of accuracy
    - The `precision` parameter does not need to be set in the config since the whole process is automatic.
    - Unfortunately, this process is currently only supported on GPU.
    - This approach is most beneficial when INT8 or FP16 exclusive quantizations (static or dynamic) are giving poor results.

All three methodolgies first pre-procsses the model before quantization adding further optimizations. This intermidate model is stored to the `pre-processed` directory. 

For this example, we will set the `precision` to `'uint8'` (since `ConvInteger` node is not currently supported for `'int8'` on ONNXRT GPU execution provider). 

We will also set the `precision_types` to `['static', 'dynamic', 'auto']` to compare all three quantization methods, whilst keeping the other settings the exact same for a fair comparison against the optimized `vgg7` model used in the previous section.


```python
JSC_TOML_PATH = "../../../machop/configs/onnx/jsc_gpu_quant.toml"
JSC_CHECKPOINT_PATH = "../../../mase_output/jsc-toy_cls_jsc/software/training_ckpts/best.ckpt"
!ch transform --config {JSC_TOML_PATH} --load {JSC_CHECKPOINT_PATH} --load-type pl
```

    [2024-03-27 20:47:18,736] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0327 20:47:20.915161 139994198443840 seed.py:54] Seed set to 0
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | Name                    |        Default         | Config. File |     Manual Override      |        Effective         |
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | task                    |     [38;5;8mclassification[0m     |     cls      |                          |           cls            |
    | load_name               |          [38;5;8mNone[0m          |              | /root/mase/mase_output/j | /root/mase/mase_output/j |
    |                         |                        |              | sc-toy_cls_jsc/software/ | sc-toy_cls_jsc/software/ |
    |                         |                        |              | training_ckpts/best.ckpt | training_ckpts/best.ckpt |
    | load_type               |           [38;5;8mmz[0m           |              |            pl            |            pl            |
    | batch_size              |          [38;5;8m128[0m           |      64      |                          |            64            |
    | to_debug                |         False          |              |                          |          False           |
    | log_level               |          info          |              |                          |           info           |
    | report_to               |      tensorboard       |              |                          |       tensorboard        |
    | seed                    |           0            |              |                          |            0             |
    | quant_config            |          None          |              |                          |           None           |
    | training_optimizer      |          adam          |              |                          |           adam           |
    | trainer_precision       |        16-mixed        |              |                          |         16-mixed         |
    | learning_rate           |         [38;5;8m1e-05[0m          |    0.001     |                          |          0.001           |
    | weight_decay            |           0            |              |                          |            0             |
    | max_epochs              |           [38;5;8m20[0m           |      10      |                          |            10            |
    | max_steps               |           -1           |              |                          |            -1            |
    | accumulate_grad_batches |           1            |              |                          |            1             |
    | log_every_n_steps       |           50           |              |                          |            50            |
    | num_workers             |           28           |              |                          |            28            |
    | num_devices             |           1            |              |                          |            1             |
    | num_nodes               |           1            |              |                          |            1             |
    | accelerator             |          [38;5;8mauto[0m          |     gpu      |                          |           gpu            |
    | strategy                |          auto          |              |                          |           auto           |
    | is_to_auto_requeue      |         False          |              |                          |          False           |
    | github_ci               |         False          |              |                          |          False           |
    | disable_dataset_cache   |         False          |              |                          |          False           |
    | target                  |  xcu250-figd2104-2L-e  |              |                          |   xcu250-figd2104-2L-e   |
    | num_targets             |          100           |              |                          |           100            |
    | is_pretrained           |         False          |              |                          |          False           |
    | max_token_len           |          512           |              |                          |           512            |
    | project_dir             | /root/mase/mase_output |              |                          |  /root/mase/mase_output  |
    | project                 |          None          |              |                          |           None           |
    | model                   |          [38;5;8mNone[0m          |   jsc-toy    |                          |         jsc-toy          |
    | dataset                 |          [38;5;8mNone[0m          |     jsc      |                          |           jsc            |
    | t_max                   |           20           |              |                          |            20            |
    | eta_min                 |         1e-06          |              |                          |          1e-06           |
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    [32mINFO    [0m [34mInitialising model 'jsc-toy'...[0m
    I0327 20:47:20.924698 139994198443840 cli.py:841] Initialising model 'jsc-toy'...
    [32mINFO    [0m [34mInitialising dataset 'jsc'...[0m
    I0327 20:47:20.926731 139994198443840 cli.py:869] Initialising dataset 'jsc'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-27[0m
    I0327 20:47:20.926998 139994198443840 cli.py:905] Project will be created at /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-27
    [32mINFO    [0m [34mTransforming model 'jsc-toy'...[0m
    I0327 20:47:21.060493 139994198443840 cli.py:365] Transforming model 'jsc-toy'...
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/jsc-toy_cls_jsc/software/training_ckpts/best.ckpt[0m
    I0327 20:47:24.415767 139994198443840 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/jsc-toy_cls_jsc/software/training_ckpts/best.ckpt
    Traceback (most recent call last):
      File "/root/mase/machop/ch", line 6, in <module>
        ChopCLI().run()
      File "/root/mase/machop/chop/cli.py", line 272, in run
        run_action_fn()
      File "/root/mase/machop/chop/cli.py", line 382, in _run_transform
        transform(**transform_params)
      File "/root/mase/machop/chop/actions/transform.py", line 80, in transform
        graph, _ = add_software_metadata_analysis_pass(graph, pass_args=None)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/root/mase/machop/chop/passes/graph/analysis/add_metadata/add_software_metadata.py", line 21, in add_software_metadata_analysis_pass
        mase_op = get_mase_op(node)
                  ^^^^^^^^^^^^^^^^^
      File "/root/mase/machop/chop/passes/graph/utils.py", line 95, in get_mase_op
        return node.meta["mase"].parameters["common"]["mase_op"]
               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
    KeyError: 'mase_op'



```python
VGG_TOML_PATH = "../../../machop/configs/onnx/vgg7_gpu_quant.toml"
VGG_CHECKPOINT_PATH = "../../../mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt"
!ch transform --config {VGG_TOML_PATH} --load {VGG_CHECKPOINT_PATH} --load-type pl
```

    [2024-03-27 17:21:25,864] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0327 17:21:28.171163 140556050732864 seed.py:54] Seed set to 0
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | Name                    |        Default         | Config. File |     Manual Override      |        Effective         |
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | task                    |     [38;5;8mclassification[0m     |     cls      |                          |           cls            |
    | load_name               |          [38;5;8mNone[0m          |              | /root/mase/mase_output/v | /root/mase/mase_output/v |
    |                         |                        |              |  gg7-pre-trained/test-   |  gg7-pre-trained/test-   |
    |                         |                        |              |     accu-0.9332.ckpt     |     accu-0.9332.ckpt     |
    | load_type               |           [38;5;8mmz[0m           |              |            pl            |            pl            |
    | batch_size              |          [38;5;8m128[0m           |      16      |                          |            16            |
    | to_debug                |         False          |              |                          |          False           |
    | log_level               |          info          |              |                          |           info           |
    | report_to               |      tensorboard       |              |                          |       tensorboard        |
    | seed                    |           0            |              |                          |            0             |
    | quant_config            |          None          |              |                          |           None           |
    | training_optimizer      |          adam          |              |                          |           adam           |
    | trainer_precision       |        16-mixed        |              |                          |         16-mixed         |
    | learning_rate           |         [38;5;8m1e-05[0m          |    0.001     |                          |          0.001           |
    | weight_decay            |           0            |              |                          |            0             |
    | max_epochs              |           [38;5;8m20[0m           |      10      |                          |            10            |
    | max_steps               |           -1           |              |                          |            -1            |
    | accumulate_grad_batches |           1            |              |                          |            1             |
    | log_every_n_steps       |           50           |              |                          |            50            |
    | num_workers             |           28           |              |                          |            28            |
    | num_devices             |           1            |              |                          |            1             |
    | num_nodes               |           1            |              |                          |            1             |
    | accelerator             |          [38;5;8mauto[0m          |     gpu      |                          |           gpu            |
    | strategy                |          auto          |              |                          |           auto           |
    | is_to_auto_requeue      |         False          |              |                          |          False           |
    | github_ci               |         False          |              |                          |          False           |
    | disable_dataset_cache   |         False          |              |                          |          False           |
    | target                  |  xcu250-figd2104-2L-e  |              |                          |   xcu250-figd2104-2L-e   |
    | num_targets             |          100           |              |                          |           100            |
    | is_pretrained           |         False          |              |                          |          False           |
    | max_token_len           |          512           |              |                          |           512            |
    | project_dir             | /root/mase/mase_output |              |                          |  /root/mase/mase_output  |
    | project                 |          None          |              |                          |           None           |
    | model                   |          [38;5;8mNone[0m          |     vgg7     |                          |           vgg7           |
    | dataset                 |          [38;5;8mNone[0m          |   cifar10    |                          |         cifar10          |
    | t_max                   |           20           |              |                          |            20            |
    | eta_min                 |         1e-06          |              |                          |          1e-06           |
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    [32mINFO    [0m [34mInitialising model 'vgg7'...[0m
    I0327 17:21:28.180522 140556050732864 cli.py:841] Initialising model 'vgg7'...
    [32mINFO    [0m [34mInitialising dataset 'cifar10'...[0m
    I0327 17:21:28.287443 140556050732864 cli.py:869] Initialising dataset 'cifar10'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-27[0m
    I0327 17:21:28.287849 140556050732864 cli.py:905] Project will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-27
    [32mINFO    [0m [34mTransforming model 'vgg7'...[0m
    I0327 17:21:28.417327 140556050732864 cli.py:365] Transforming model 'vgg7'...
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0327 17:21:34.383455 140556050732864 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0327 17:21:46.570539 140556050732864 onnx_runtime.py:48] Converting PyTorch model to ONNX...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-27[0m
    I0327 17:21:46.571265 140556050732864 onnx_runtime.py:50] Project will be created at /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-27
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-27/optimized/version_28/model.onnx[0m
    I0327 17:21:51.925619 140556050732864 onnx_runtime.py:68] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-27/optimized/version_28/model.onnx
    [32mINFO    [0m [34mONNX Model Summary: 
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+
    | Index |            Name            |   Type   |                                Inputs                               |               Outputs               |                     Attributes                    |
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+
    |   0   |   /feature_layers.0/Conv   |   Conv   |                 input, onnx::Conv_78, onnx::Conv_79                 |   /feature_layers.0/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   1   |   /feature_layers.2/Relu   |   Relu   |                   /feature_layers.0/Conv_output_0                   |   /feature_layers.2/Relu_output_0   |                                                   |
    |   2   |   /feature_layers.3/Conv   |   Conv   |    /feature_layers.2/Relu_output_0, onnx::Conv_81, onnx::Conv_82    |   /feature_layers.3/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   3   |   /feature_layers.5/Relu   |   Relu   |                   /feature_layers.3/Conv_output_0                   |   /feature_layers.5/Relu_output_0   |                                                   |
    |   4   | /feature_layers.6/MaxPool  | MaxPool  |                   /feature_layers.5/Relu_output_0                   |  /feature_layers.6/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   5   |   /feature_layers.7/Conv   |   Conv   |   /feature_layers.6/MaxPool_output_0, onnx::Conv_84, onnx::Conv_85  |   /feature_layers.7/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   6   |   /feature_layers.9/Relu   |   Relu   |                   /feature_layers.7/Conv_output_0                   |   /feature_layers.9/Relu_output_0   |                                                   |
    |   7   |  /feature_layers.10/Conv   |   Conv   |    /feature_layers.9/Relu_output_0, onnx::Conv_87, onnx::Conv_88    |   /feature_layers.10/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   8   |  /feature_layers.12/Relu   |   Relu   |                   /feature_layers.10/Conv_output_0                  |   /feature_layers.12/Relu_output_0  |                                                   |
    |   9   | /feature_layers.13/MaxPool | MaxPool  |                   /feature_layers.12/Relu_output_0                  | /feature_layers.13/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   10  |  /feature_layers.14/Conv   |   Conv   |  /feature_layers.13/MaxPool_output_0, onnx::Conv_90, onnx::Conv_91  |   /feature_layers.14/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   11  |  /feature_layers.16/Relu   |   Relu   |                   /feature_layers.14/Conv_output_0                  |   /feature_layers.16/Relu_output_0  |                                                   |
    |   12  |  /feature_layers.17/Conv   |   Conv   |    /feature_layers.16/Relu_output_0, onnx::Conv_93, onnx::Conv_94   |   /feature_layers.17/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   13  |  /feature_layers.19/Relu   |   Relu   |                   /feature_layers.17/Conv_output_0                  |   /feature_layers.19/Relu_output_0  |                                                   |
    |   14  | /feature_layers.20/MaxPool | MaxPool  |                   /feature_layers.19/Relu_output_0                  | /feature_layers.20/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   15  |         /Constant          | Constant |                                                                     |          /Constant_output_0         |                       value                       |
    |   16  |          /Reshape          | Reshape  |       /feature_layers.20/MaxPool_output_0, /Constant_output_0       |          /Reshape_output_0          |                                                   |
    |   17  |     /classifier.0/Gemm     |   Gemm   |      /Reshape_output_0, classifier.0.weight, classifier.0.bias      |     /classifier.0/Gemm_output_0     |                alpha, beta, transB                |
    |   18  |     /classifier.1/Relu     |   Relu   |                     /classifier.0/Gemm_output_0                     |     /classifier.1/Relu_output_0     |                                                   |
    |   19  |     /classifier.2/Gemm     |   Gemm   | /classifier.1/Relu_output_0, classifier.2.weight, classifier.2.bias |     /classifier.2/Gemm_output_0     |                alpha, beta, transB                |
    |   20  |     /classifier.3/Relu     |   Relu   |                     /classifier.2/Gemm_output_0                     |     /classifier.3/Relu_output_0     |                                                   |
    |   21  |      /last_layer/Gemm      |   Gemm   |   /classifier.3/Relu_output_0, last_layer.weight, last_layer.bias   |                  76                 |                alpha, beta, transB                |
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+[0m
    I0327 17:21:52.048147 140556050732864 onnx_runtime.py:90] ONNX Model Summary: 
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+
    | Index |            Name            |   Type   |                                Inputs                               |               Outputs               |                     Attributes                    |
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+
    |   0   |   /feature_layers.0/Conv   |   Conv   |                 input, onnx::Conv_78, onnx::Conv_79                 |   /feature_layers.0/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   1   |   /feature_layers.2/Relu   |   Relu   |                   /feature_layers.0/Conv_output_0                   |   /feature_layers.2/Relu_output_0   |                                                   |
    |   2   |   /feature_layers.3/Conv   |   Conv   |    /feature_layers.2/Relu_output_0, onnx::Conv_81, onnx::Conv_82    |   /feature_layers.3/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   3   |   /feature_layers.5/Relu   |   Relu   |                   /feature_layers.3/Conv_output_0                   |   /feature_layers.5/Relu_output_0   |                                                   |
    |   4   | /feature_layers.6/MaxPool  | MaxPool  |                   /feature_layers.5/Relu_output_0                   |  /feature_layers.6/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   5   |   /feature_layers.7/Conv   |   Conv   |   /feature_layers.6/MaxPool_output_0, onnx::Conv_84, onnx::Conv_85  |   /feature_layers.7/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   6   |   /feature_layers.9/Relu   |   Relu   |                   /feature_layers.7/Conv_output_0                   |   /feature_layers.9/Relu_output_0   |                                                   |
    |   7   |  /feature_layers.10/Conv   |   Conv   |    /feature_layers.9/Relu_output_0, onnx::Conv_87, onnx::Conv_88    |   /feature_layers.10/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   8   |  /feature_layers.12/Relu   |   Relu   |                   /feature_layers.10/Conv_output_0                  |   /feature_layers.12/Relu_output_0  |                                                   |
    |   9   | /feature_layers.13/MaxPool | MaxPool  |                   /feature_layers.12/Relu_output_0                  | /feature_layers.13/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   10  |  /feature_layers.14/Conv   |   Conv   |  /feature_layers.13/MaxPool_output_0, onnx::Conv_90, onnx::Conv_91  |   /feature_layers.14/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   11  |  /feature_layers.16/Relu   |   Relu   |                   /feature_layers.14/Conv_output_0                  |   /feature_layers.16/Relu_output_0  |                                                   |
    |   12  |  /feature_layers.17/Conv   |   Conv   |    /feature_layers.16/Relu_output_0, onnx::Conv_93, onnx::Conv_94   |   /feature_layers.17/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   13  |  /feature_layers.19/Relu   |   Relu   |                   /feature_layers.17/Conv_output_0                  |   /feature_layers.19/Relu_output_0  |                                                   |
    |   14  | /feature_layers.20/MaxPool | MaxPool  |                   /feature_layers.19/Relu_output_0                  | /feature_layers.20/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   15  |         /Constant          | Constant |                                                                     |          /Constant_output_0         |                       value                       |
    |   16  |          /Reshape          | Reshape  |       /feature_layers.20/MaxPool_output_0, /Constant_output_0       |          /Reshape_output_0          |                                                   |
    |   17  |     /classifier.0/Gemm     |   Gemm   |      /Reshape_output_0, classifier.0.weight, classifier.0.bias      |     /classifier.0/Gemm_output_0     |                alpha, beta, transB                |
    |   18  |     /classifier.1/Relu     |   Relu   |                     /classifier.0/Gemm_output_0                     |     /classifier.1/Relu_output_0     |                                                   |
    |   19  |     /classifier.2/Gemm     |   Gemm   | /classifier.1/Relu_output_0, classifier.2.weight, classifier.2.bias |     /classifier.2/Gemm_output_0     |                alpha, beta, transB                |
    |   20  |     /classifier.3/Relu     |   Relu   |                     /classifier.2/Gemm_output_0                     |     /classifier.3/Relu_output_0     |                                                   |
    |   21  |      /last_layer/Gemm      |   Gemm   |   /classifier.3/Relu_output_0, last_layer.weight, last_layer.bias   |                  76                 |                alpha, beta, transB                |
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+
    [32mINFO    [0m [34mQuantizing model using dynamic quantization...[0m
    I0327 17:21:52.830610 140556050732864 quantize.py:33] Quantizing model using dynamic quantization...
    [32mINFO    [0m [34mQuantization complete. Model is now dynamically quantized.[0m
    I0327 17:21:53.849616 140556050732864 quantize.py:48] Quantization complete. Model is now dynamically quantized.
    [32mINFO    [0m [34mPerforming runtime analysis on original graph...[0m
    I0327 17:21:53.849973 140556050732864 transform.py:170] Performing runtime analysis on original graph...
    [32mINFO    [0m [34mStarting transformation analysis on vgg7[0m
    I0327 17:21:53.850163 140556050732864 analysis.py:276] Starting transformation analysis on vgg7
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.89024    |
    |      Average Precision       |   0.92059    |
    |        Average Recall        |   0.92039    |
    |       Average F1 Score       |    0.9203    |
    |         Average Loss         |   0.22849    |
    |       Average Latency        |  3.8322 ms   |
    |   Average GPU Power Usage    |   27.99 W    |
    | Inference Energy Consumption | 0.029796 mWh |
    +------------------------------+--------------+[0m
    I0327 17:21:59.743183 140556050732864 analysis.py:404] 
    Results vgg7:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.89024    |
    |      Average Precision       |   0.92059    |
    |        Average Recall        |   0.92039    |
    |       Average F1 Score       |    0.9203    |
    |         Average Loss         |   0.22849    |
    |       Average Latency        |  3.8322 ms   |
    |   Average GPU Power Usage    |   27.99 W    |
    | Inference Energy Consumption | 0.029796 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/mase_graph/version_30/model.json[0m
    I0327 17:21:59.744388 140556050732864 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/mase_graph/version_30/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on onnx-optimized graph...[0m
    I0327 17:21:59.744547 140556050732864 transform.py:176] Performing runtime analysis on onnx-optimized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0327 17:21:59.744721 140556050732864 analysis.py:65] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    2024-03-27 17:21:59.744998725 [I:onnxruntime:, inference_session.cc:514 TraceSessionOptions] Session Options {  execution_mode:0 execution_order:DEFAULT enable_profiling:0 optimized_model_filepath: enable_mem_pattern:1 enable_mem_reuse:1 enable_cpu_mem_arena:1 profile_file_prefix:onnxruntime_profile_ session_logid: session_log_severity_level:1 session_log_verbosity_level:0 max_num_graph_transformation_steps:10 graph_optimization_level:3 intra_op_param:OrtThreadPoolParams { thread_pool_size: 0 auto_set_affinity: 0 allow_spinning: 1 dynamic_block_base_: 0 stack_size: 0 affinity_str:  set_denormal_as_zero: 0 } inter_op_param:OrtThreadPoolParams { thread_pool_size: 0 auto_set_affinity: 0 allow_spinning: 1 dynamic_block_base_: 0 stack_size: 0 affinity_str:  set_denormal_as_zero: 0 } use_per_session_threads:1 thread_pool_allow_spinning:1 use_deterministic_compute:0 config_options: {  } }
    2024-03-27 17:21:59.745028243 [I:onnxruntime:, inference_session.cc:422 ConstructorCommon] Creating and using per session threadpools since use_per_session_threads_ is true
    2024-03-27 17:21:59.745038118 [I:onnxruntime:, inference_session.cc:440 ConstructorCommon] Dynamic block base set to 0
    2024-03-27 17:22:00.385208051 [I:onnxruntime:, inference_session.cc:1583 Initialize] Initializing session.
    2024-03-27 17:22:00.385235625 [I:onnxruntime:, inference_session.cc:1620 Initialize] Adding default CPU execution provider.
    2024-03-27 17:22:00.385419189 [I:onnxruntime:, graph_partitioner.cc:902 InlineFunctionsAOT] This model does not have any local functions defined. AOT Inlining is not performed
    2024-03-27 17:22:00.385466555 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.385502303 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.385519893 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:00.385598388 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:00.385975467 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:00.386003825 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:00.386020162 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:00.386038044 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.386048586 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.386069470 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.386086884 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.386102688 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:00.386146922 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer TransposeOptimizer modified: 0 with status: OK
    2024-03-27 17:22:00.386175004 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.386190918 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:00.386232102 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:00.386516198 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:00.386541521 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:00.386558136 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:00.386575422 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.386585666 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.386606223 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.386623615 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.386639108 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:00.386665391 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.386681224 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:00.386719253 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:00.386991967 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:00.387017372 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:00.387034051 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:00.387050955 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.387061197 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.387081646 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.387098812 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.387114440 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:00.387140526 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.387156282 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:00.387193483 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:00.387463704 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:00.387489044 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:00.387505798 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:00.387522926 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.387533237 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.387554017 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.387571313 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.387586866 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:00.387613102 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.387629107 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:00.387666130 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:00.387933015 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:00.387958364 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:00.387975177 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:00.387992509 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.388002887 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.388023359 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.388040971 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.388056601 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:00.388082584 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.388098575 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:00.388134429 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:00.388401578 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:00.388427069 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:00.388444152 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:00.388461112 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.388471620 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.388492328 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.388509739 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.388525709 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:00.388552213 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.388568269 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:00.388604727 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:00.388870900 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:00.388895718 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:00.388912653 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:00.388929717 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.388940001 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.388960929 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.388978521 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.388994341 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:00.389020783 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.389036763 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:00.389073300 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:00.389377452 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:00.389404385 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:00.389421257 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:00.389438386 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.389448856 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.389469390 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.389486930 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.389502647 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:00.389529280 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.389545627 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:00.389584231 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:00.389855438 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:00.389880560 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:00.389897168 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:00.389914170 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.389924429 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.389945087 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.389962424 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.389978417 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:00.390004430 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.390020410 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:00.390057540 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:00.390322400 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:00.390347056 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:00.390364462 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:00.390381621 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.390391840 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.390412459 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.390429907 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:00.390445570 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:00.390764898 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer TransposeOptimizer_CPUExecutionProvider modified: 0 with status: OK
    2024-03-27 17:22:00.390785064 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.390801589 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.390819763 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.390836845 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:00.390853542 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:00.390953179 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 1 with status: OK
    2024-03-27 17:22:00.391246436 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391267104 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391282941 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391297581 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391312522 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391331919 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391347181 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391363675 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391378845 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391394728 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:00.391411396 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391428035 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391443602 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391459651 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391477599 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391492575 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391508722 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.391523998 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.391538958 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.391554282 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391569505 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391584534 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391603834 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391618721 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391633555 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391648256 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391662618 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391677152 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391695298 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391710551 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391726436 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391741134 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391755949 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:00.391772045 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391787669 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391802603 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391817291 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391833884 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391848727 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391864565 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.391879692 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.391894182 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.391909159 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391924104 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391939204 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391957563 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391972281 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.391987227 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392001893 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392015825 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392030397 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392048416 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392063232 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392079125 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392093889 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392108528 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:00.392124779 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392140472 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392155218 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392170225 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392186802 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392201354 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392217163 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.392231459 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.392245817 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.392261084 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392276007 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392290863 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392309183 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392324332 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392338998 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392353849 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392367850 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392382212 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392400141 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392414875 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392430618 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392445510 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392460140 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:00.392476160 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392491974 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392506781 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392521263 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392537916 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392552387 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392567988 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.392582425 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.392596756 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.392611558 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392626592 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392641379 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392659423 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392674319 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392689032 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392703638 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392717572 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392731966 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392749674 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392764763 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392780348 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392794958 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392809866 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:00.392825705 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392841530 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392856366 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392871034 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392887340 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392902124 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392917787 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.392931917 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.392946469 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.392961476 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392976561 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:00.392991571 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393009811 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393024430 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393039278 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393054070 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393067802 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393082352 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393100038 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393114788 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393165848 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393183631 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393198172 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:00.393214317 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393230230 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393244846 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393259482 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393276112 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393290698 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393306269 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.393320665 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.393334905 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.393350216 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393370065 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393384849 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393403266 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393418196 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393432704 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393447805 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393461657 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393476031 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393494605 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393509373 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393524838 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393539643 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393554139 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:00.393569920 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393585683 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393600315 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393614893 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393631519 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393645910 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393661275 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.393675787 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.393690010 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.393704932 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393719810 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393734499 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393752313 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393767368 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393781843 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393796374 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393810414 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393824760 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393842198 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393856838 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393872539 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393887045 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393901891 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:00.393917651 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393933189 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393947921 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393962491 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393978739 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:00.393993365 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394008816 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.394023118 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.394037678 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.394052693 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394067399 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394082089 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394100242 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394114894 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394129762 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394144183 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394157714 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394172355 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394190014 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394204865 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394220624 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394235304 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394249958 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:00.394265993 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394281612 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394296400 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394311293 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394327784 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394342244 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394357907 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.394372281 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.394386591 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.394401638 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394416748 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394431541 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394449821 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394464553 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394479272 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394494060 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394507906 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394522308 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394540034 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394554640 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394570240 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394585012 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394599608 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:00.394615451 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394631470 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394646328 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394661035 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394677688 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394692234 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394707597 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.394723076 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer NchwcTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.394745648 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer NhwcTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.394760237 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvAddActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:00.394780018 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RemoveDuplicateCastTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.394789993 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CastFloat16Transformer modified: 0 with status: OK
    2024-03-27 17:22:00.395029114 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MemcpyTransformer modified: 0 with status: OK
    2024-03-27 17:22:00.523888071 [I:onnxruntime:, allocation_planner.cc:2432 CreateGraphPartitioner] Use DeviceBasedPartition as default
    2024-03-27 17:22:00.524156067 [I:onnxruntime:, session_state_utils.cc:201 SaveInitializedTensors] Saving initialized tensors.
    2024-03-27 17:22:00.565000154 [I:onnxruntime:, session_state_utils.cc:345 SaveInitializedTensors] Done saving initialized tensors
    2024-03-27 17:22:00.565345605 [I:onnxruntime:, inference_session.cc:1969 Initialize] Session successfully initialized.
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0327 17:22:00.565490 140556050732864 analysis.py:276] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.88533    |
    |      Average Precision       |   0.91987    |
    |        Average Recall        |   0.91579    |
    |       Average F1 Score       |   0.91576    |
    |         Average Loss         |   0.24644    |
    |       Average Latency        |  2.2131 ms   |
    |   Average GPU Power Usage    |   34.238 W   |
    | Inference Energy Consumption | 0.021048 mWh |
    +------------------------------+--------------+[0m
    I0327 17:22:06.356243 140556050732864 analysis.py:404] 
    Results vgg7-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.88533    |
    |      Average Precision       |   0.91987    |
    |        Average Recall        |   0.91579    |
    |       Average F1 Score       |   0.91576    |
    |         Average Loss         |   0.24644    |
    |       Average Latency        |  2.2131 ms   |
    |   Average GPU Power Usage    |   34.238 W   |
    | Inference Energy Consumption | 0.021048 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/onnx/version_35/model.json[0m
    I0327 17:22:06.357525 140556050732864 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/onnx/version_35/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on dynamic quantized graph...[0m
    I0327 17:22:06.365460 140556050732864 transform.py:196] Performing runtime analysis on dynamic quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0327 17:22:06.365708 140556050732864 analysis.py:65] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    2024-03-27 17:22:06.365941569 [I:onnxruntime:, inference_session.cc:514 TraceSessionOptions] Session Options {  execution_mode:0 execution_order:DEFAULT enable_profiling:0 optimized_model_filepath: enable_mem_pattern:1 enable_mem_reuse:1 enable_cpu_mem_arena:1 profile_file_prefix:onnxruntime_profile_ session_logid: session_log_severity_level:1 session_log_verbosity_level:0 max_num_graph_transformation_steps:10 graph_optimization_level:3 intra_op_param:OrtThreadPoolParams { thread_pool_size: 0 auto_set_affinity: 0 allow_spinning: 1 dynamic_block_base_: 0 stack_size: 0 affinity_str:  set_denormal_as_zero: 0 } inter_op_param:OrtThreadPoolParams { thread_pool_size: 0 auto_set_affinity: 0 allow_spinning: 1 dynamic_block_base_: 0 stack_size: 0 affinity_str:  set_denormal_as_zero: 0 } use_per_session_threads:1 thread_pool_allow_spinning:1 use_deterministic_compute:0 config_options: {  } }
    2024-03-27 17:22:06.365966284 [I:onnxruntime:, inference_session.cc:422 ConstructorCommon] Creating and using per session threadpools since use_per_session_threads_ is true
    2024-03-27 17:22:06.365976349 [I:onnxruntime:, inference_session.cc:440 ConstructorCommon] Dynamic block base set to 0
    2024-03-27 17:22:06.426553879 [I:onnxruntime:, inference_session.cc:1583 Initialize] Initializing session.
    2024-03-27 17:22:06.426580219 [I:onnxruntime:, inference_session.cc:1620 Initialize] Adding default CPU execution provider.
    2024-03-27 17:22:06.426740534 [I:onnxruntime:, graph_partitioner.cc:902 InlineFunctionsAOT] This model does not have any local functions defined. AOT Inlining is not performed
    2024-03-27 17:22:06.426801301 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.426890270 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.426926364 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:06.427095615 [I:onnxruntime:, constant_sharing.cc:256 ApplyImpl] Total shared scalar initializer count: 5
    2024-03-27 17:22:06.427111647 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:06.427988531 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:06.428559919 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 1 with status: OK
    2024-03-27 17:22:06.429217839 [I:onnxruntime:, graph.cc:3596 CleanUnusedInitializersAndNodeArgs] Removing initializer 'ortshared_7_1_4_0'. It is no longer used by any node.
    2024-03-27 17:22:06.429233747 [I:onnxruntime:, graph.cc:3596 CleanUnusedInitializersAndNodeArgs] Removing initializer 'onnx::Conv_94'. It is no longer used by any node.
    2024-03-27 17:22:06.429243823 [I:onnxruntime:, graph.cc:3596 CleanUnusedInitializersAndNodeArgs] Removing initializer 'onnx::Conv_91'. It is no longer used by any node.
    2024-03-27 17:22:06.429253170 [I:onnxruntime:, graph.cc:3596 CleanUnusedInitializersAndNodeArgs] Removing initializer 'onnx::Conv_82'. It is no longer used by any node.
    2024-03-27 17:22:06.429263756 [I:onnxruntime:, graph.cc:3596 CleanUnusedInitializersAndNodeArgs] Removing initializer 'onnx::Conv_79'. It is no longer used by any node.
    2024-03-27 17:22:06.429272950 [I:onnxruntime:, graph.cc:3596 CleanUnusedInitializersAndNodeArgs] Removing initializer 'onnx::Conv_85'. It is no longer used by any node.
    2024-03-27 17:22:06.429282623 [I:onnxruntime:, graph.cc:3596 CleanUnusedInitializersAndNodeArgs] Removing initializer 'onnx::Conv_88'. It is no longer used by any node.
    2024-03-27 17:22:06.429346830 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:06.429381444 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.429392225 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.429439545 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.429475092 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.429504695 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:06.429615965 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer TransposeOptimizer modified: 0 with status: OK
    2024-03-27 17:22:06.429691140 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.429723887 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:06.429857746 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:06.430636316 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:06.430697216 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:06.430730721 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:06.430762101 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.430772120 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.430818532 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.430853515 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.430882080 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:06.430956027 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.430988121 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:06.431122323 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:06.431895293 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:06.431955467 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:06.431989079 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:06.432020827 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.432030913 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.432077189 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.432112371 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.432141163 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:06.432214150 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.432246750 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:06.432378033 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:06.433100160 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:06.433156668 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:06.433179576 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:06.433198587 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.433205061 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.433230363 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.433250429 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.433267405 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:06.433315802 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.433336651 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:06.433427526 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:06.433947468 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:06.433987252 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:06.434007368 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:06.434025879 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.434032781 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.434058099 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.434077949 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.434095197 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:06.434141970 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.434162535 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:06.434271871 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:06.434782925 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:06.434822640 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:06.434842865 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:06.434861249 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.434867694 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.434893022 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.434912905 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.434929948 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:06.434977088 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.434997440 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:06.435085504 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:06.435596844 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:06.435636446 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:06.435656527 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:06.435675326 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.435681708 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.435706800 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.435726540 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.435743577 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:06.435789626 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.435810023 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:06.435902151 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:06.436417654 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:06.436457575 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:06.436477786 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:06.436496342 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.436503154 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.436528289 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.436548169 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.436565382 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:06.436611667 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.436632025 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:06.436720836 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:06.437262645 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:06.437303755 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:06.437324568 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:06.437343370 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.437349748 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.437375437 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.437395464 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.437412510 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:06.437460357 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer Level1_RuleBasedTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.437480903 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DoubleQDQPairsRemover modified: 0 with status: OK
    2024-03-27 17:22:06.437570533 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantSharing modified: 1 with status: OK
    2024-03-27 17:22:06.438085513 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CommonSubexpressionElimination modified: 0 with status: OK
    2024-03-27 17:22:06.438125976 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConstantFolding modified: 0 with status: OK
    2024-03-27 17:22:06.438146192 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulAddFusion modified: 0 with status: OK
    2024-03-27 17:22:06.438165050 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ReshapeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.438171551 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FreeDimensionOverrideTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.438196932 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQPropagationTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.438217227 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EnsureUniqueDQForNodeUnit modified: 0 with status: OK
    2024-03-27 17:22:06.438234362 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RocmBlasAltImpl modified: 0 with status: OK
    2024-03-27 17:22:06.438258910 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: DynamicQuantizeLinear node name: input_QuantizeLinear
    2024-03-27 17:22:06.438269327 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: ConvInteger node name: /feature_layers.0/Conv_quant
    2024-03-27 17:22:06.438351790 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: DynamicQuantizeLinear node name: /feature_layers.2/Relu_output_0_QuantizeLinear
    2024-03-27 17:22:06.438359583 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: ConvInteger node name: /feature_layers.3/Conv_quant
    2024-03-27 17:22:06.438421317 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: DynamicQuantizeLinear node name: /feature_layers.6/MaxPool_output_0_QuantizeLinear
    2024-03-27 17:22:06.438428367 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: ConvInteger node name: /feature_layers.7/Conv_quant
    2024-03-27 17:22:06.438478439 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: DynamicQuantizeLinear node name: /feature_layers.9/Relu_output_0_QuantizeLinear
    2024-03-27 17:22:06.438484944 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: ConvInteger node name: /feature_layers.10/Conv_quant
    2024-03-27 17:22:06.438541275 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: DynamicQuantizeLinear node name: /feature_layers.13/MaxPool_output_0_QuantizeLinear
    2024-03-27 17:22:06.438547880 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: ConvInteger node name: /feature_layers.14/Conv_quant
    2024-03-27 17:22:06.438597321 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: DynamicQuantizeLinear node name: /feature_layers.16/Relu_output_0_QuantizeLinear
    2024-03-27 17:22:06.438613839 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: ConvInteger node name: /feature_layers.17/Conv_quant
    2024-03-27 17:22:06.438672617 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: DynamicQuantizeLinear node name: /Reshape_output_0_QuantizeLinear
    2024-03-27 17:22:06.438686755 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: MatMulInteger node name: /classifier.0/Gemm_MatMul_quant
    2024-03-27 17:22:06.438737028 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: DynamicQuantizeLinear node name: /classifier.1/Relu_output_0_QuantizeLinear
    2024-03-27 17:22:06.438746708 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: MatMulInteger node name: /classifier.2/Gemm_MatMul_quant
    2024-03-27 17:22:06.438796582 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: DynamicQuantizeLinear node name: /classifier.3/Relu_output_0_QuantizeLinear
    2024-03-27 17:22:06.438805967 [I:onnxruntime:, cuda_execution_provider.cc:2397 GetCapability] CUDA kernel not found in registries for Op type: MatMulInteger node name: /last_layer/Gemm_MatMul_quant
    2024-03-27 17:22:06.439395725 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer TransposeOptimizer_CPUExecutionProvider modified: 0 with status: OK
    2024-03-27 17:22:06.439420704 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.439442220 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.439461613 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439482420 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439501294 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439525137 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439544767 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439563452 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439582560 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439600227 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439618299 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439643789 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439663445 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439684094 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439705978 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439726818 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:06.439766662 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439802481 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439827432 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439848946 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439873106 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439892154 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439913557 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.439933231 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.439953725 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.439973265 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.439992918 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440011679 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440034907 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440054096 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440073216 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440092350 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440109199 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440127191 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440151302 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440170155 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440190660 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440211057 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440229670 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:06.440265337 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440303439 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440326630 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440347768 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440371761 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440390283 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440411202 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.440430595 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.440450761 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.440469941 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440489724 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440508265 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440531700 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440551272 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440569811 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440589130 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440606138 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440624167 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440648891 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440668024 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440688391 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440709053 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440729044 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:06.440764023 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440800844 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440823697 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440844230 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440868241 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440886737 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440907277 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.440927026 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.440947219 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.440966066 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.440986821 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:06.441005132 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:06.441028323 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.441047912 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.441066533 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.441085441 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.441102589 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:06.441120767 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.521494382 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:06.521561274 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:06.521616828 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.521675682 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.521728118 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:06.521829329 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.521916125 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.521947148 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:06.521974291 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522005092 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522028203 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522054092 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.522078679 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.522104993 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.522128994 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522154119 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522177343 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522206675 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522230290 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522254408 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522278011 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522299409 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522322394 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522352659 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522376319 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522401915 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522427631 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522452438 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:06.522497101 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522538752 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522567622 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522594151 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522623657 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522646648 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522672698 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.522697297 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.522722057 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.522746166 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522770931 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522793935 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522823125 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522846791 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522869704 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522893582 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522914567 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522936743 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522967206 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:06.522990921 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523016445 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523042295 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523067024 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:06.523110464 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523151871 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523180348 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523206302 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523236579 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523259311 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523284778 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.523309460 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.523334295 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.523357572 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523382875 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523405981 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523434509 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523458750 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523481651 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523505086 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523526236 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523548926 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523578336 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523602255 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523627315 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523652428 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523677471 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:06.523720303 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523760809 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523789919 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523815749 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523845293 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523868276 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523893792 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.523918324 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.523943367 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.523967532 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.523991881 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524015259 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524044049 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524067738 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524091183 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524114649 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524135333 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524157929 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524186890 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524210332 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524235663 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524261173 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524285741 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:06.524329196 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524369579 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524397958 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524424256 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524453690 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524476241 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524502139 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.524526448 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.524551025 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.524574840 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524599481 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524622186 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524651195 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524674952 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524699232 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524723070 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524744112 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524765923 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524795141 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524818975 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524844194 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524870008 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524894853 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:06.524937491 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.524978916 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525007397 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525033176 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525062945 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525085825 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525111331 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.525154866 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQS8ToU8Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.525182454 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQSelectorActionTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.525210453 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GemmActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525235456 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulIntegerToFloatFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525258426 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer DynamicQuantizeMatMulFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525287535 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525311830 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525335477 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer LayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525358801 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SimplifiedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525379952 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer AttentionFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525402121 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer EmbedLayerNormFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525432508 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSplitFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525456477 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GatherToSliceFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525482063 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatmulTransposeFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525507938 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525532115 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer SkipLayerNormFusion modified: 1 with status: OK
    2024-03-27 17:22:06.525576925 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer FastGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525618082 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QuickGeluFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525647218 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasSoftmaxFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525673159 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer BiasDropoutFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525702686 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulScaleFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525725763 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MatMulActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525751409 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer QDQFinalCleanupTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.525781819 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer NchwcTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.525834055 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer NhwcTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.525861998 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer ConvAddActivationFusion modified: 0 with status: OK
    2024-03-27 17:22:06.525930287 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer RemoveDuplicateCastTransformer modified: 0 with status: OK
    2024-03-27 17:22:06.525940219 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer CastFloat16Transformer modified: 0 with status: OK
    2024-03-27 17:22:06.527609614 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /Reshape_output_0_scale for CUDAExecutionProvider
    2024-03-27 17:22:06.527630635 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /classifier.0/Gemm_output_0_MatMul_output_quantized for CUDAExecutionProvider
    2024-03-27 17:22:06.527650126 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /classifier.1/Relu_output_0_scale for CUDAExecutionProvider
    2024-03-27 17:22:06.527663459 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /classifier.2/Gemm_output_0_MatMul_output_quantized for CUDAExecutionProvider
    2024-03-27 17:22:06.527677195 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /classifier.3/Relu_output_0_scale for CUDAExecutionProvider
    2024-03-27 17:22:06.527689993 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.0/Conv_output_0_output_quantized for CUDAExecutionProvider
    2024-03-27 17:22:06.527703428 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.10/Conv_output_0_output_quantized for CUDAExecutionProvider
    2024-03-27 17:22:06.527716217 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.13/MaxPool_output_0_scale for CUDAExecutionProvider
    2024-03-27 17:22:06.527729369 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.14/Conv_output_0_output_quantized for CUDAExecutionProvider
    2024-03-27 17:22:06.527741553 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.16/Relu_output_0_scale for CUDAExecutionProvider
    2024-03-27 17:22:06.527755164 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.17/Conv_output_0_output_quantized for CUDAExecutionProvider
    2024-03-27 17:22:06.527767700 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.2/Relu_output_0_scale for CUDAExecutionProvider
    2024-03-27 17:22:06.527781360 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.3/Conv_output_0_output_quantized for CUDAExecutionProvider
    2024-03-27 17:22:06.527794147 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.6/MaxPool_output_0_scale for CUDAExecutionProvider
    2024-03-27 17:22:06.527806961 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.7/Conv_output_0_output_quantized for CUDAExecutionProvider
    2024-03-27 17:22:06.527818853 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after /feature_layers.9/Relu_output_0_scale for CUDAExecutionProvider
    2024-03-27 17:22:06.527832406 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after 76_MatMul_output_quantized for CUDAExecutionProvider
    2024-03-27 17:22:06.527844719 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyFromHost after input_scale for CUDAExecutionProvider
    2024-03-27 17:22:06.527858291 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyToHost before /Reshape_output_0 for CUDAExecutionProvider
    2024-03-27 17:22:06.527871679 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyToHost before /classifier.1/Relu_output_0 for CUDAExecutionProvider
    2024-03-27 17:22:06.527884435 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyToHost before /classifier.3/Relu_output_0 for CUDAExecutionProvider
    2024-03-27 17:22:06.527898611 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyToHost before /feature_layers.13/MaxPool_output_0 for CUDAExecutionProvider
    2024-03-27 17:22:06.527922932 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyToHost before /feature_layers.16/Relu_output_0 for CUDAExecutionProvider
    2024-03-27 17:22:06.527936632 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyToHost before /feature_layers.2/Relu_output_0 for CUDAExecutionProvider
    2024-03-27 17:22:06.527949500 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyToHost before /feature_layers.6/MaxPool_output_0 for CUDAExecutionProvider
    2024-03-27 17:22:06.527963024 [I:onnxruntime:, transformer_memcpy.cc:329 AddCopyNode] Add MemcpyToHost before /feature_layers.9/Relu_output_0 for CUDAExecutionProvider
    [0;93m2024-03-27 17:22:06.527975105 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 26 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.[m
    2024-03-27 17:22:06.527996554 [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer MemcpyTransformer modified: 1 with status: OK
    [0;93m2024-03-27 17:22:06.528851394 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
    [0;93m2024-03-27 17:22:06.528865962 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
    2024-03-27 17:22:06.531018699 [I:onnxruntime:, allocation_planner.cc:2432 CreateGraphPartitioner] Use DeviceBasedPartition as default
    2024-03-27 17:22:06.531765153 [I:onnxruntime:, session_state_utils.cc:201 SaveInitializedTensors] Saving initialized tensors.
    2024-03-27 17:22:06.538411512 [I:onnxruntime:, session_state_utils.cc:345 SaveInitializedTensors] Done saving initialized tensors
    2024-03-27 17:22:06.542799432 [I:onnxruntime:, inference_session.cc:1969 Initialize] Session successfully initialized.
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0327 17:22:06.542922 140556050732864 analysis.py:276] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+-----------+
    |      Metric (Per Batch)      |   Value   |
    +------------------------------+-----------+
    |    Average Test Accuracy     |  0.88349  |
    |      Average Precision       |   0.918   |
    |        Average Recall        |  0.91382  |
    |       Average F1 Score       |  0.91367  |
    |         Average Loss         |  0.25111  |
    |       Average Latency        | 382.94 ms |
    |   Average GPU Power Usage    |  22.28 W  |
    | Inference Energy Consumption | 2.37 mWh  |
    +------------------------------+-----------+[0m
    I0327 17:22:52.726737 140556050732864 analysis.py:404] 
    Results vgg7-onnx:
    +------------------------------+-----------+
    |      Metric (Per Batch)      |   Value   |
    +------------------------------+-----------+
    |    Average Test Accuracy     |  0.88349  |
    |      Average Precision       |   0.918   |
    |        Average Recall        |  0.91382  |
    |       Average F1 Score       |  0.91367  |
    |         Average Loss         |  0.25111  |
    |       Average Latency        | 382.94 ms |
    |   Average GPU Power Usage    |  22.28 W  |
    | Inference Energy Consumption | 2.37 mWh  |
    +------------------------------+-----------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/onnx/version_36/model.json[0m
    I0327 17:22:52.728345 140556050732864 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/onnx/version_36/model.json
    [32mINFO    [0m [34mSaved mase graph to /root/mase/mase_output/vgg7_cls_cifar10_2024-03-27/software/transform/transformed_ckpt[0m
    I0327 17:24:53.698515 140556050732864 save_and_load.py:147] Saved mase graph to /root/mase/mase_output/vgg7_cls_cifar10_2024-03-27/software/transform/transformed_ckpt
    [32mINFO    [0m [34mTransformation is completed[0m
    I0327 17:24:53.699082 140556050732864 cli.py:383] Transformation is completed



```python
VGG_TOML_PATH = "../../../machop/configs/onnx/vgg7_cpu_quant.toml"
VGG_CHECKPOINT_PATH = "../../../mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt"
!ch transform --config {VGG_TOML_PATH} --load {VGG_CHECKPOINT_PATH} --load-type pl 
```

    [2024-03-27 17:15:55,643] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0327 17:15:58.257988 139658009347904 seed.py:54] Seed set to 0
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | Name                    |        Default         | Config. File |     Manual Override      |        Effective         |
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | task                    |     [38;5;8mclassification[0m     |     cls      |                          |           cls            |
    | load_name               |          [38;5;8mNone[0m          |              | /root/mase/mase_output/v | /root/mase/mase_output/v |
    |                         |                        |              |  gg7-pre-trained/test-   |  gg7-pre-trained/test-   |
    |                         |                        |              |     accu-0.9332.ckpt     |     accu-0.9332.ckpt     |
    | load_type               |           [38;5;8mmz[0m           |              |            pl            |            pl            |
    | batch_size              |          [38;5;8m128[0m           |      16      |                          |            16            |
    | to_debug                |         False          |              |                          |          False           |
    | log_level               |          info          |              |                          |           info           |
    | report_to               |      tensorboard       |              |                          |       tensorboard        |
    | seed                    |           0            |              |                          |            0             |
    | quant_config            |          None          |              |                          |           None           |
    | training_optimizer      |          adam          |              |                          |           adam           |
    | trainer_precision       |        16-mixed        |              |                          |         16-mixed         |
    | learning_rate           |         [38;5;8m1e-05[0m          |    0.001     |                          |          0.001           |
    | weight_decay            |           0            |              |                          |            0             |
    | max_epochs              |           [38;5;8m20[0m           |      10      |                          |            10            |
    | max_steps               |           -1           |              |                          |            -1            |
    | accumulate_grad_batches |           1            |              |                          |            1             |
    | log_every_n_steps       |           50           |              |                          |            50            |
    | num_workers             |           28           |              |                          |            28            |
    | num_devices             |           1            |              |                          |            1             |
    | num_nodes               |           1            |              |                          |            1             |
    | accelerator             |          [38;5;8mauto[0m          |     cpu      |                          |           cpu            |
    | strategy                |          auto          |              |                          |           auto           |
    | is_to_auto_requeue      |         False          |              |                          |          False           |
    | github_ci               |         False          |              |                          |          False           |
    | disable_dataset_cache   |         False          |              |                          |          False           |
    | target                  |  xcu250-figd2104-2L-e  |              |                          |   xcu250-figd2104-2L-e   |
    | num_targets             |          100           |              |                          |           100            |
    | is_pretrained           |         False          |              |                          |          False           |
    | max_token_len           |          512           |              |                          |           512            |
    | project_dir             | /root/mase/mase_output |              |                          |  /root/mase/mase_output  |
    | project                 |          None          |              |                          |           None           |
    | model                   |          [38;5;8mNone[0m          |     vgg7     |                          |           vgg7           |
    | dataset                 |          [38;5;8mNone[0m          |   cifar10    |                          |         cifar10          |
    | t_max                   |           20           |              |                          |            20            |
    | eta_min                 |         1e-06          |              |                          |          1e-06           |
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    [32mINFO    [0m [34mInitialising model 'vgg7'...[0m
    I0327 17:15:58.267456 139658009347904 cli.py:841] Initialising model 'vgg7'...
    [32mINFO    [0m [34mInitialising dataset 'cifar10'...[0m
    I0327 17:15:58.376318 139658009347904 cli.py:869] Initialising dataset 'cifar10'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-27[0m
    I0327 17:15:58.376691 139658009347904 cli.py:905] Project will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-27
    [32mINFO    [0m [34mTransforming model 'vgg7'...[0m
    I0327 17:15:58.507813 139658009347904 cli.py:365] Transforming model 'vgg7'...
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0327 17:16:04.670324 139658009347904 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0327 17:16:21.516777 139658009347904 onnx_runtime.py:48] Converting PyTorch model to ONNX...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-27[0m
    I0327 17:16:21.517552 139658009347904 onnx_runtime.py:50] Project will be created at /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-27
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-27/optimized/version_27/model.onnx[0m
    I0327 17:16:30.050149 139658009347904 onnx_runtime.py:68] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-27/optimized/version_27/model.onnx
    [32mINFO    [0m [34mONNX Model Summary: 
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+
    | Index |            Name            |   Type   |                                Inputs                               |               Outputs               |                     Attributes                    |
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+
    |   0   |   /feature_layers.0/Conv   |   Conv   |                 input, onnx::Conv_78, onnx::Conv_79                 |   /feature_layers.0/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   1   |   /feature_layers.2/Relu   |   Relu   |                   /feature_layers.0/Conv_output_0                   |   /feature_layers.2/Relu_output_0   |                                                   |
    |   2   |   /feature_layers.3/Conv   |   Conv   |    /feature_layers.2/Relu_output_0, onnx::Conv_81, onnx::Conv_82    |   /feature_layers.3/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   3   |   /feature_layers.5/Relu   |   Relu   |                   /feature_layers.3/Conv_output_0                   |   /feature_layers.5/Relu_output_0   |                                                   |
    |   4   | /feature_layers.6/MaxPool  | MaxPool  |                   /feature_layers.5/Relu_output_0                   |  /feature_layers.6/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   5   |   /feature_layers.7/Conv   |   Conv   |   /feature_layers.6/MaxPool_output_0, onnx::Conv_84, onnx::Conv_85  |   /feature_layers.7/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   6   |   /feature_layers.9/Relu   |   Relu   |                   /feature_layers.7/Conv_output_0                   |   /feature_layers.9/Relu_output_0   |                                                   |
    |   7   |  /feature_layers.10/Conv   |   Conv   |    /feature_layers.9/Relu_output_0, onnx::Conv_87, onnx::Conv_88    |   /feature_layers.10/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   8   |  /feature_layers.12/Relu   |   Relu   |                   /feature_layers.10/Conv_output_0                  |   /feature_layers.12/Relu_output_0  |                                                   |
    |   9   | /feature_layers.13/MaxPool | MaxPool  |                   /feature_layers.12/Relu_output_0                  | /feature_layers.13/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   10  |  /feature_layers.14/Conv   |   Conv   |  /feature_layers.13/MaxPool_output_0, onnx::Conv_90, onnx::Conv_91  |   /feature_layers.14/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   11  |  /feature_layers.16/Relu   |   Relu   |                   /feature_layers.14/Conv_output_0                  |   /feature_layers.16/Relu_output_0  |                                                   |
    |   12  |  /feature_layers.17/Conv   |   Conv   |    /feature_layers.16/Relu_output_0, onnx::Conv_93, onnx::Conv_94   |   /feature_layers.17/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   13  |  /feature_layers.19/Relu   |   Relu   |                   /feature_layers.17/Conv_output_0                  |   /feature_layers.19/Relu_output_0  |                                                   |
    |   14  | /feature_layers.20/MaxPool | MaxPool  |                   /feature_layers.19/Relu_output_0                  | /feature_layers.20/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   15  |         /Constant          | Constant |                                                                     |          /Constant_output_0         |                       value                       |
    |   16  |          /Reshape          | Reshape  |       /feature_layers.20/MaxPool_output_0, /Constant_output_0       |          /Reshape_output_0          |                                                   |
    |   17  |     /classifier.0/Gemm     |   Gemm   |      /Reshape_output_0, classifier.0.weight, classifier.0.bias      |     /classifier.0/Gemm_output_0     |                alpha, beta, transB                |
    |   18  |     /classifier.1/Relu     |   Relu   |                     /classifier.0/Gemm_output_0                     |     /classifier.1/Relu_output_0     |                                                   |
    |   19  |     /classifier.2/Gemm     |   Gemm   | /classifier.1/Relu_output_0, classifier.2.weight, classifier.2.bias |     /classifier.2/Gemm_output_0     |                alpha, beta, transB                |
    |   20  |     /classifier.3/Relu     |   Relu   |                     /classifier.2/Gemm_output_0                     |     /classifier.3/Relu_output_0     |                                                   |
    |   21  |      /last_layer/Gemm      |   Gemm   |   /classifier.3/Relu_output_0, last_layer.weight, last_layer.bias   |                  76                 |                alpha, beta, transB                |
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+[0m
    I0327 17:16:30.143252 139658009347904 onnx_runtime.py:90] ONNX Model Summary: 
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+
    | Index |            Name            |   Type   |                                Inputs                               |               Outputs               |                     Attributes                    |
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+
    |   0   |   /feature_layers.0/Conv   |   Conv   |                 input, onnx::Conv_78, onnx::Conv_79                 |   /feature_layers.0/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   1   |   /feature_layers.2/Relu   |   Relu   |                   /feature_layers.0/Conv_output_0                   |   /feature_layers.2/Relu_output_0   |                                                   |
    |   2   |   /feature_layers.3/Conv   |   Conv   |    /feature_layers.2/Relu_output_0, onnx::Conv_81, onnx::Conv_82    |   /feature_layers.3/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   3   |   /feature_layers.5/Relu   |   Relu   |                   /feature_layers.3/Conv_output_0                   |   /feature_layers.5/Relu_output_0   |                                                   |
    |   4   | /feature_layers.6/MaxPool  | MaxPool  |                   /feature_layers.5/Relu_output_0                   |  /feature_layers.6/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   5   |   /feature_layers.7/Conv   |   Conv   |   /feature_layers.6/MaxPool_output_0, onnx::Conv_84, onnx::Conv_85  |   /feature_layers.7/Conv_output_0   |   dilations, group, kernel_shape, pads, strides   |
    |   6   |   /feature_layers.9/Relu   |   Relu   |                   /feature_layers.7/Conv_output_0                   |   /feature_layers.9/Relu_output_0   |                                                   |
    |   7   |  /feature_layers.10/Conv   |   Conv   |    /feature_layers.9/Relu_output_0, onnx::Conv_87, onnx::Conv_88    |   /feature_layers.10/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   8   |  /feature_layers.12/Relu   |   Relu   |                   /feature_layers.10/Conv_output_0                  |   /feature_layers.12/Relu_output_0  |                                                   |
    |   9   | /feature_layers.13/MaxPool | MaxPool  |                   /feature_layers.12/Relu_output_0                  | /feature_layers.13/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   10  |  /feature_layers.14/Conv   |   Conv   |  /feature_layers.13/MaxPool_output_0, onnx::Conv_90, onnx::Conv_91  |   /feature_layers.14/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   11  |  /feature_layers.16/Relu   |   Relu   |                   /feature_layers.14/Conv_output_0                  |   /feature_layers.16/Relu_output_0  |                                                   |
    |   12  |  /feature_layers.17/Conv   |   Conv   |    /feature_layers.16/Relu_output_0, onnx::Conv_93, onnx::Conv_94   |   /feature_layers.17/Conv_output_0  |   dilations, group, kernel_shape, pads, strides   |
    |   13  |  /feature_layers.19/Relu   |   Relu   |                   /feature_layers.17/Conv_output_0                  |   /feature_layers.19/Relu_output_0  |                                                   |
    |   14  | /feature_layers.20/MaxPool | MaxPool  |                   /feature_layers.19/Relu_output_0                  | /feature_layers.20/MaxPool_output_0 | ceil_mode, dilations, kernel_shape, pads, strides |
    |   15  |         /Constant          | Constant |                                                                     |          /Constant_output_0         |                       value                       |
    |   16  |          /Reshape          | Reshape  |       /feature_layers.20/MaxPool_output_0, /Constant_output_0       |          /Reshape_output_0          |                                                   |
    |   17  |     /classifier.0/Gemm     |   Gemm   |      /Reshape_output_0, classifier.0.weight, classifier.0.bias      |     /classifier.0/Gemm_output_0     |                alpha, beta, transB                |
    |   18  |     /classifier.1/Relu     |   Relu   |                     /classifier.0/Gemm_output_0                     |     /classifier.1/Relu_output_0     |                                                   |
    |   19  |     /classifier.2/Gemm     |   Gemm   | /classifier.1/Relu_output_0, classifier.2.weight, classifier.2.bias |     /classifier.2/Gemm_output_0     |                alpha, beta, transB                |
    |   20  |     /classifier.3/Relu     |   Relu   |                     /classifier.2/Gemm_output_0                     |     /classifier.3/Relu_output_0     |                                                   |
    |   21  |      /last_layer/Gemm      |   Gemm   |   /classifier.3/Relu_output_0, last_layer.weight, last_layer.bias   |                  76                 |                alpha, beta, transB                |
    +-------+----------------------------+----------+---------------------------------------------------------------------+-------------------------------------+---------------------------------------------------+
    [32mINFO    [0m [34mQuantizing model using dynamic quantization...[0m
    I0327 17:16:30.866059 139658009347904 quantize.py:33] Quantizing model using dynamic quantization...
    [32mINFO    [0m [34mQuantization complete. Model is now dynamically quantized.[0m
    I0327 17:16:31.844397 139658009347904 quantize.py:48] Quantization complete. Model is now dynamically quantized.
    [32mINFO    [0m [34mPerforming runtime analysis on original graph...[0m
    I0327 17:16:31.844740 139658009347904 transform.py:170] Performing runtime analysis on original graph...
    [32mINFO    [0m [34mStarting transformation analysis on vgg7[0m
    I0327 17:16:31.844887 139658009347904 analysis.py:267] Starting transformation analysis on vgg7
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.89024   |
    |      Average Precision       |   0.92059   |
    |        Average Recall        |   0.92039   |
    |       Average F1 Score       |   0.9203    |
    |         Average Loss         |   0.22849   |
    |       Average Latency        |  248.33 ms  |
    |   Average GPU Power Usage    |   10.6 W    |
    | Inference Energy Consumption | 0.73121 mWh |
    +------------------------------+-------------+[0m
    I0327 17:17:06.058737 139658009347904 analysis.py:395] 
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.89024   |
    |      Average Precision       |   0.92059   |
    |        Average Recall        |   0.92039   |
    |       Average F1 Score       |   0.9203    |
    |         Average Loss         |   0.22849   |
    |       Average Latency        |  248.33 ms  |
    |   Average GPU Power Usage    |   10.6 W    |
    | Inference Energy Consumption | 0.73121 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/mase_graph/version_29/model.json[0m
    I0327 17:17:06.059940 139658009347904 analysis.py:81] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/mase_graph/version_29/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on onnx-optimized graph...[0m
    I0327 17:17:06.060117 139658009347904 transform.py:176] Performing runtime analysis on onnx-optimized graph...
    [32mINFO    [0m [34mUsing ['CPUExecutionProvider'] as ONNX execution provider.[0m
    I0327 17:17:06.060289 139658009347904 analysis.py:65] Using ['CPUExecutionProvider'] as ONNX execution provider.
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0327 17:17:07.697271 139658009347904 analysis.py:267] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.88533   |
    |      Average Precision       |   0.91987   |
    |        Average Recall        |   0.91579   |
    |       Average F1 Score       |   0.91576   |
    |         Average Loss         |   0.24645   |
    |       Average Latency        |  164.19 ms  |
    |   Average GPU Power Usage    |  10.605 W   |
    | Inference Energy Consumption | 0.48368 mWh |
    +------------------------------+-------------+[0m
    I0327 17:17:33.687082 139658009347904 analysis.py:395] 
    Results vgg7-onnx:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.88533   |
    |      Average Precision       |   0.91987   |
    |        Average Recall        |   0.91579   |
    |       Average F1 Score       |   0.91576   |
    |         Average Loss         |   0.24645   |
    |       Average Latency        |  164.19 ms  |
    |   Average GPU Power Usage    |  10.605 W   |
    | Inference Energy Consumption | 0.48368 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/onnx/version_33/model.json[0m
    I0327 17:17:33.689377 139658009347904 analysis.py:81] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/onnx/version_33/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on dynamic quantized graph...[0m
    I0327 17:17:33.702023 139658009347904 transform.py:196] Performing runtime analysis on dynamic quantized graph...
    [32mINFO    [0m [34mUsing ['CPUExecutionProvider'] as ONNX execution provider.[0m
    I0327 17:17:33.702450 139658009347904 analysis.py:65] Using ['CPUExecutionProvider'] as ONNX execution provider.
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0327 17:17:33.757479 139658009347904 analysis.py:267] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+------------+
    |      Metric (Per Batch)      |   Value    |
    +------------------------------+------------+
    |    Average Test Accuracy     |   0.8837   |
    |      Average Precision       |   0.9179   |
    |        Average Recall        |  0.91382   |
    |       Average F1 Score       |  0.91368   |
    |         Average Loss         |  0.25096   |
    |       Average Latency        | 373.42 ms  |
    |   Average GPU Power Usage    |  10.607 W  |
    | Inference Energy Consumption | 1.1003 mWh |
    +------------------------------+------------+[0m
    I0327 17:18:21.187844 139658009347904 analysis.py:395] 
    Results vgg7-onnx:
    +------------------------------+------------+
    |      Metric (Per Batch)      |   Value    |
    +------------------------------+------------+
    |    Average Test Accuracy     |   0.8837   |
    |      Average Precision       |   0.9179   |
    |        Average Recall        |  0.91382   |
    |       Average F1 Score       |  0.91368   |
    |         Average Loss         |  0.25096   |
    |       Average Latency        | 373.42 ms  |
    |   Average GPU Power Usage    |  10.607 W  |
    | Inference Energy Consumption | 1.1003 mWh |
    +------------------------------+------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/onnx/version_34/model.json[0m
    I0327 17:18:21.196517 139658009347904 analysis.py:81] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-27/onnx/version_34/model.json
    [32mINFO    [0m [34mSaved mase graph to /root/mase/mase_output/vgg7_cls_cifar10_2024-03-27/software/transform/transformed_ckpt[0m
    I0327 17:19:55.452324 139658009347904 save_and_load.py:147] Saved mase graph to /root/mase/mase_output/vgg7_cls_cifar10_2024-03-27/software/transform/transformed_ckpt
    [32mINFO    [0m [34mTransformation is completed[0m
    I0327 17:19:55.452704 139658009347904 cli.py:383] Transformation is completed



```python
MOBILENET_SMALL_TOML_PATH = "../../../machop/configs/onnx/mobilenetv3_small_gpu_quant.toml"
MOBILENET_SMALL_CHECKPOINT_PATH = "../../../mase_output/mobilenetv3-small_pre-trained_mnist/mobilenetv3_small_mnist_0.997.ckpt"

!ch transform --config {MOBILENET_SMALL_TOML_PATH} --load {MOBILENET_SMALL_CHECKPOINT_PATH} --load-type pl
```

    [2024-03-28 00:06:42,363] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0328 00:06:45.331041 140494178068288 seed.py:54] Seed set to 0
    +-------------------------+------------------------+-------------------+--------------------------+--------------------------+
    | Name                    |        Default         |   Config. File    |     Manual Override      |        Effective         |
    +-------------------------+------------------------+-------------------+--------------------------+--------------------------+
    | task                    |     [38;5;8mclassification[0m     |        cls        |                          |           cls            |
    | load_name               |          [38;5;8mNone[0m          |                   | /root/mase/mase_output/m | /root/mase/mase_output/m |
    |                         |                        |                   | obilenetv3-small_pre-tra | obilenetv3-small_pre-tra |
    |                         |                        |                   | ined_mnist/mobilenetv3_s | ined_mnist/mobilenetv3_s |
    |                         |                        |                   |  mall_mnist_0.997.ckpt   |  mall_mnist_0.997.ckpt   |
    | load_type               |           [38;5;8mmz[0m           |                   |            pl            |            pl            |
    | batch_size              |          [38;5;8m128[0m           |        64         |                          |            64            |
    | to_debug                |         False          |                   |                          |          False           |
    | log_level               |          info          |                   |                          |           info           |
    | report_to               |      tensorboard       |                   |                          |       tensorboard        |
    | seed                    |           0            |                   |                          |            0             |
    | quant_config            |          None          |                   |                          |           None           |
    | training_optimizer      |          adam          |                   |                          |           adam           |
    | trainer_precision       |        16-mixed        |                   |                          |         16-mixed         |
    | learning_rate           |         [38;5;8m1e-05[0m          |       0.001       |                          |          0.001           |
    | weight_decay            |           0            |                   |                          |            0             |
    | max_epochs              |           [38;5;8m20[0m           |        10         |                          |            10            |
    | max_steps               |           -1           |                   |                          |            -1            |
    | accumulate_grad_batches |           1            |                   |                          |            1             |
    | log_every_n_steps       |           50           |                   |                          |            50            |
    | num_workers             |           28           |                   |                          |            28            |
    | num_devices             |           1            |                   |                          |            1             |
    | num_nodes               |           1            |                   |                          |            1             |
    | accelerator             |          [38;5;8mauto[0m          |        gpu        |                          |           gpu            |
    | strategy                |          auto          |                   |                          |           auto           |
    | is_to_auto_requeue      |         False          |                   |                          |          False           |
    | github_ci               |         False          |                   |                          |          False           |
    | disable_dataset_cache   |         False          |                   |                          |          False           |
    | target                  |  xcu250-figd2104-2L-e  |                   |                          |   xcu250-figd2104-2L-e   |
    | num_targets             |          100           |                   |                          |           100            |
    | is_pretrained           |         [38;5;8mFalse[0m          |                   |           True           |           True           |
    | max_token_len           |          512           |                   |                          |           512            |
    | project_dir             | /root/mase/mase_output |                   |                          |  /root/mase/mase_output  |
    | project                 |          None          |                   |                          |           None           |
    | model                   |          [38;5;8mNone[0m          | mobilenetv3_small |                          |    mobilenetv3_small     |
    | dataset                 |          [38;5;8mNone[0m          |       mnist       |                          |          mnist           |
    | t_max                   |           20           |                   |                          |            20            |
    | eta_min                 |         1e-06          |                   |                          |          1e-06           |
    +-------------------------+------------------------+-------------------+--------------------------+--------------------------+
    [32mINFO    [0m [34mInitialising model 'mobilenetv3_small'...[0m
    I0328 00:06:45.340413 140494178068288 cli.py:841] Initialising model 'mobilenetv3_small'...
    [33mWARNING [0m [34mThe num_classes(=10) != 1000. The last layer (classifier.3) is random initialized[0m
    W0328 00:06:45.393730 140494178068288 mobilenetv3.py:486] The num_classes(=10) != 1000. The last layer (classifier.3) is random initialized
    [32mINFO    [0m [34mPretrained weights loaded into MobileNetV3[0m
    I0328 00:06:45.400723 140494178068288 mobilenetv3.py:490] Pretrained weights loaded into MobileNetV3
    [32mINFO    [0m [34mInitialising dataset 'mnist'...[0m
    I0328 00:06:45.401582 140494178068288 cli.py:869] Initialising dataset 'mnist'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/mobilenetv3_small_cls_mnist_2024-03-28[0m
    I0328 00:06:45.401916 140494178068288 cli.py:905] Project will be created at /root/mase/mase_output/mobilenetv3_small_cls_mnist_2024-03-28
    [32mINFO    [0m [34mTransforming model 'mobilenetv3_small'...[0m
    I0328 00:06:45.517536 140494178068288 cli.py:365] Transforming model 'mobilenetv3_small'...
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/mobilenetv3-small_pre-trained_mnist/mobilenetv3_small_mnist_0.997.ckpt[0m
    I0328 00:06:45.874669 140494178068288 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/mobilenetv3-small_pre-trained_mnist/mobilenetv3_small_mnist_0.997.ckpt
    Traceback (most recent call last):
      File "/root/mase/machop/ch", line 6, in <module>
        ChopCLI().run()
      File "/root/mase/machop/chop/cli.py", line 272, in run
        run_action_fn()
      File "/root/mase/machop/chop/cli.py", line 382, in _run_transform
        transform(**transform_params)
      File "/root/mase/machop/chop/actions/transform.py", line 77, in transform
        graph, _ = add_common_metadata_analysis_pass(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/root/mase/machop/chop/passes/graph/analysis/add_metadata/add_common_metadata.py", line 412, in add_common_metadata_analysis_pass
        graph = graph_iterator_for_metadata(graph, **pass_args)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/root/mase/machop/chop/passes/graph/analysis/add_metadata/add_common_metadata.py", line 187, in graph_iterator_for_metadata
        result = dummy_in[node.name]
                 ~~~~~~~~^^^^^^^^^^^
    KeyError: 'input_1'



```python
MOBILENET_TOML_PATH = "../../../machop/configs/onnx/mobilenetv3_large_gpu_quant.toml"
!ch transform --config {MOBILENET_TOML_PATH}
```

    [2024-03-27 21:02:08,767] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0327 21:02:10.969612 139809641850688 seed.py:54] Seed set to 0
    +-------------------------+------------------------+-------------------+-----------------+------------------------+
    | Name                    |        Default         |   Config. File    | Manual Override |       Effective        |
    +-------------------------+------------------------+-------------------+-----------------+------------------------+
    | task                    |     [38;5;8mclassification[0m     |        cls        |                 |          cls           |
    | load_name               |          None          |                   |                 |          None          |
    | load_type               |           mz           |                   |                 |           mz           |
    | batch_size              |          [38;5;8m128[0m           |        64         |                 |           64           |
    | to_debug                |         False          |                   |                 |         False          |
    | log_level               |          info          |                   |                 |          info          |
    | report_to               |      tensorboard       |                   |                 |      tensorboard       |
    | seed                    |           0            |                   |                 |           0            |
    | quant_config            |          None          |                   |                 |          None          |
    | training_optimizer      |          adam          |                   |                 |          adam          |
    | trainer_precision       |        16-mixed        |                   |                 |        16-mixed        |
    | learning_rate           |         [38;5;8m1e-05[0m          |       0.001       |                 |         0.001          |
    | weight_decay            |           0            |                   |                 |           0            |
    | max_epochs              |           [38;5;8m20[0m           |        10         |                 |           10           |
    | max_steps               |           -1           |                   |                 |           -1           |
    | accumulate_grad_batches |           1            |                   |                 |           1            |
    | log_every_n_steps       |           50           |                   |                 |           50           |
    | num_workers             |           28           |                   |                 |           28           |
    | num_devices             |           1            |                   |                 |           1            |
    | num_nodes               |           1            |                   |                 |           1            |
    | accelerator             |          [38;5;8mauto[0m          |        gpu        |                 |          gpu           |
    | strategy                |          auto          |                   |                 |          auto          |
    | is_to_auto_requeue      |         False          |                   |                 |         False          |
    | github_ci               |         False          |                   |                 |         False          |
    | disable_dataset_cache   |         False          |                   |                 |         False          |
    | target                  |  xcu250-figd2104-2L-e  |                   |                 |  xcu250-figd2104-2L-e  |
    | num_targets             |          100           |                   |                 |          100           |
    | is_pretrained           |         False          |                   |                 |         False          |
    | max_token_len           |          512           |                   |                 |          512           |
    | project_dir             | /root/mase/mase_output |                   |                 | /root/mase/mase_output |
    | project                 |          None          |                   |                 |          None          |
    | model                   |          [38;5;8mNone[0m          | mobilenetv3_large |                 |   mobilenetv3_large    |
    | dataset                 |          [38;5;8mNone[0m          |       mnist       |                 |         mnist          |
    | t_max                   |           20           |                   |                 |           20           |
    | eta_min                 |         1e-06          |                   |                 |         1e-06          |
    +-------------------------+------------------------+-------------------+-----------------+------------------------+
    [32mINFO    [0m [34mInitialising model 'mobilenetv3_large'...[0m
    I0327 21:02:10.977810 139809641850688 cli.py:841] Initialising model 'mobilenetv3_large'...
    [32mINFO    [0m [34mMobileNetV3 randomly initialized[0m
    I0327 21:02:11.054991 139809641850688 mobilenetv3.py:492] MobileNetV3 randomly initialized
    [32mINFO    [0m [34mInitialising dataset 'mnist'...[0m
    I0327 21:02:11.055421 139809641850688 cli.py:869] Initialising dataset 'mnist'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/mobilenetv3_large_cls_mnist_2024-03-27[0m
    I0327 21:02:11.055759 139809641850688 cli.py:905] Project will be created at /root/mase/mase_output/mobilenetv3_large_cls_mnist_2024-03-27
    [32mINFO    [0m [34mTransforming model 'mobilenetv3_large'...[0m
    I0327 21:02:11.183370 139809641850688 cli.py:365] Transforming model 'mobilenetv3_large'...
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0327 21:02:29.136042 139809641850688 onnx_runtime.py:48] Converting PyTorch model to ONNX...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/onnxrt/mobilenetv3_large_cls_mnist_2024-03-27[0m
    I0327 21:02:29.136666 139809641850688 onnx_runtime.py:50] Project will be created at /root/mase/mase_output/onnxrt/mobilenetv3_large_cls_mnist_2024-03-27
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/mobilenetv3_large_cls_mnist_2024-03-27/optimized/version_1/model.onnx[0m
    I0327 21:02:33.051671 139809641850688 onnx_runtime.py:68] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/mobilenetv3_large_cls_mnist_2024-03-27/optimized/version_1/model.onnx
    [32mINFO    [0m [34mONNX Model Summary: 
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+
    | Index |               Name              |        Type       |                                                  Inputs                                                 |                 Outputs                  |                   Attributes                  |
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+
    |   0   |            Identity_0           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_660              |                                               |
    |   1   |            Identity_1           |      Identity     |                                              onnx::Conv_639                                             |              onnx::Conv_657              |                                               |
    |   2   |            Identity_2           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_654              |                                               |
    |   3   |            Identity_3           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_651              |                                               |
    |   4   |            Identity_4           |      Identity     |                                              onnx::Conv_639                                             |              onnx::Conv_648              |                                               |
    |   5   |            Identity_5           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_645              |                                               |
    |   6   |            Identity_6           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_642              |                                               |
    |   7   |            Identity_7           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_636              |                                               |
    |   8   |            Identity_8           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_633              |                                               |
    |   9   |            Identity_9           |      Identity     |                                              onnx::Conv_621                                             |              onnx::Conv_630              |                                               |
    |   10  |           Identity_10           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_627              |                                               |
    |   11  |           Identity_11           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_624              |                                               |
    |   12  |           Identity_12           |      Identity     |                                      1.features.11.block.2.fc2.bias                                     |              onnx::Conv_618              |                                               |
    |   13  |           Identity_13           |      Identity     |                                      1.features.11.block.2.fc2.bias                                     |              onnx::Conv_615              |                                               |
    |   14  |           Identity_14           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_612              |                                               |
    |   15  |           Identity_15           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_609              |                                               |
    |   16  |           Identity_16           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_606              |                                               |
    |   17  |           Identity_17           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_603              |                                               |
    |   18  |           Identity_18           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_600              |                                               |
    |   19  |           Identity_19           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_594              |                                               |
    |   20  |           Identity_20           |      Identity     |                                              onnx::Conv_588                                             |              onnx::Conv_591              |                                               |
    |   21  |           Identity_21           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |              onnx::Conv_582              |                                               |
    |   22  |           Identity_22           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |              onnx::Conv_579              |                                               |
    |   23  |           Identity_23           |      Identity     |                                              onnx::Conv_558                                             |              onnx::Conv_576              |                                               |
    |   24  |           Identity_24           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_573              |                                               |
    |   25  |           Identity_25           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_570              |                                               |
    |   26  |           Identity_26           |      Identity     |                                              onnx::Conv_558                                             |              onnx::Conv_567              |                                               |
    |   27  |           Identity_27           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_564              |                                               |
    |   28  |           Identity_28           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_561              |                                               |
    |   29  |           Identity_29           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_555              |                                               |
    |   30  |           Identity_30           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_552              |                                               |
    |   31  |           Identity_31           |      Identity     |                                      1.features.4.block.2.fc1.bias                                      |              onnx::Conv_549              |                                               |
    |   32  |           Identity_32           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_546              |                                               |
    |   33  |           Identity_33           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_543              |                                               |
    |   34  |           Identity_34           |      Identity     |                                      1.features.4.block.2.fc1.bias                                      |              onnx::Conv_540              |                                               |
    |   35  |           Identity_35           |      Identity     |                                              onnx::Conv_534                                             |              onnx::Conv_537              |                                               |
    |   36  |           Identity_36           |      Identity     |                                              onnx::Conv_525                                             |              onnx::Conv_531              |                                               |
    |   37  |           Identity_37           |      Identity     |                                              onnx::Conv_525                                             |              onnx::Conv_528              |                                               |
    |   38  |           Identity_38           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |      1.features.15.block.2.fc2.bias      |                                               |
    |   39  |           Identity_39           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |      1.features.15.block.2.fc1.bias      |                                               |
    |   40  |           Identity_40           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |      1.features.13.block.2.fc2.bias      |                                               |
    |   41  |           Identity_41           |      Identity     |                                      1.features.12.block.2.fc1.bias                                     |      1.features.13.block.2.fc1.bias      |                                               |
    |   42  |           Identity_42           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |      1.features.11.block.2.fc1.bias      |                                               |
    |   43  |           Identity_43           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |      1.features.6.block.2.fc2.bias       |                                               |
    |   44  |           Identity_44           |      Identity     |                                      1.features.5.block.2.fc1.bias                                      |      1.features.6.block.2.fc1.bias       |                                               |
    |   45  |             /0/Conv             |        Conv       |                                         input, 0.weight, 0.bias                                         |             /0/Conv_output_0             | dilations, group, kernel_shape, pads, strides |
    |   46  |        /features.0.0/Conv       |        Conv       |                             /0/Conv_output_0, onnx::Conv_524, onnx::Conv_525                            |       /features.0.0/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   47  |    /features.0.2/HardSigmoid    |    HardSigmoid    |                                       /features.0.0/Conv_output_0                                       |    /features.0.2/HardSigmoid_output_0    |                     alpha                     |
    |   48  |        /features.0.2/Mul        |        Mul        |                     /features.0.0/Conv_output_0, /features.0.2/HardSigmoid_output_0                     |        /features.0.2/Mul_output_0        |                                               |
    |   49  |         /block.0.0/Conv         |        Conv       |                        /features.0.2/Mul_output_0, onnx::Conv_527, onnx::Conv_528                       |         /block.0.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   50  |         /block.0.2/Relu         |        Relu       |                                         /block.0.0/Conv_output_0                                        |         /block.0.2/Relu_output_0         |                                               |
    |   51  |         /block.1.0/Conv         |        Conv       |                         /block.0.2/Relu_output_0, onnx::Conv_530, onnx::Conv_531                        |         /block.1.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   52  |               /Add              |        Add        |                           /block.1.0/Conv_output_0, /features.0.2/Mul_output_0                          |              /Add_output_0               |                                               |
    |   53  |        /block.0.0_1/Conv        |        Conv       |                              /Add_output_0, onnx::Conv_533, onnx::Conv_534                              |        /block.0.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   54  |        /block.0.2_1/Relu        |        Relu       |                                        /block.0.0_1/Conv_output_0                                       |        /block.0.2_1/Relu_output_0        |                                               |
    |   55  |        /block.1.0_1/Conv        |        Conv       |                        /block.0.2_1/Relu_output_0, onnx::Conv_536, onnx::Conv_537                       |        /block.1.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   56  |         /block.1.2/Relu         |        Relu       |                                        /block.1.0_1/Conv_output_0                                       |         /block.1.2/Relu_output_0         |                                               |
    |   57  |         /block.2.0/Conv         |        Conv       |                         /block.1.2/Relu_output_0, onnx::Conv_539, onnx::Conv_540                        |         /block.2.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   58  |        /block.0.0_2/Conv        |        Conv       |                         /block.2.0/Conv_output_0, onnx::Conv_542, onnx::Conv_543                        |        /block.0.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   59  |        /block.0.2_2/Relu        |        Relu       |                                        /block.0.0_2/Conv_output_0                                       |        /block.0.2_2/Relu_output_0        |                                               |
    |   60  |        /block.1.0_2/Conv        |        Conv       |                        /block.0.2_2/Relu_output_0, onnx::Conv_545, onnx::Conv_546                       |        /block.1.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   61  |        /block.1.2_1/Relu        |        Relu       |                                        /block.1.0_2/Conv_output_0                                       |        /block.1.2_1/Relu_output_0        |                                               |
    |   62  |        /block.2.0_1/Conv        |        Conv       |                        /block.1.2_1/Relu_output_0, onnx::Conv_548, onnx::Conv_549                       |        /block.2.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   63  |              /Add_1             |        Add        |                           /block.2.0_1/Conv_output_0, /block.2.0/Conv_output_0                          |             /Add_1_output_0              |                                               |
    |   64  |        /block.0.0_3/Conv        |        Conv       |                             /Add_1_output_0, onnx::Conv_551, onnx::Conv_552                             |        /block.0.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   65  |        /block.0.2_3/Relu        |        Relu       |                                        /block.0.0_3/Conv_output_0                                       |        /block.0.2_3/Relu_output_0        |                                               |
    |   66  |        /block.1.0_3/Conv        |        Conv       |                        /block.0.2_3/Relu_output_0, onnx::Conv_554, onnx::Conv_555                       |        /block.1.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   67  |        /block.1.2_2/Relu        |        Relu       |                                        /block.1.0_3/Conv_output_0                                       |        /block.1.2_2/Relu_output_0        |                                               |
    |   68  |    /avgpool/GlobalAveragePool   | GlobalAveragePool |                                        /block.1.2_2/Relu_output_0                                       |   /avgpool/GlobalAveragePool_output_0    |                                               |
    |   69  |            /fc1/Conv            |        Conv       |   /avgpool/GlobalAveragePool_output_0, 1.features.4.block.2.fc1.weight, 1.features.4.block.2.fc1.bias   |            /fc1/Conv_output_0            | dilations, group, kernel_shape, pads, strides |
    |   70  |         /activation/Relu        |        Relu       |                                            /fc1/Conv_output_0                                           |        /activation/Relu_output_0         |                                               |
    |   71  |            /fc2/Conv            |        Conv       |        /activation/Relu_output_0, 1.features.4.block.2.fc2.weight, 1.features.4.block.2.fc2.bias        |            /fc2/Conv_output_0            | dilations, group, kernel_shape, pads, strides |
    |   72  |  /scale_activation/HardSigmoid  |    HardSigmoid    |                                            /fc2/Conv_output_0                                           |  /scale_activation/HardSigmoid_output_0  |                     alpha                     |
    |   73  |               /Mul              |        Mul        |                    /scale_activation/HardSigmoid_output_0, /block.1.2_2/Relu_output_0                   |              /Mul_output_0               |                                               |
    |   74  |         /block.3.0/Conv         |        Conv       |                              /Mul_output_0, onnx::Conv_557, onnx::Conv_558                              |         /block.3.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   75  |        /block.0.0_4/Conv        |        Conv       |                         /block.3.0/Conv_output_0, onnx::Conv_560, onnx::Conv_561                        |        /block.0.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   76  |        /block.0.2_4/Relu        |        Relu       |                                        /block.0.0_4/Conv_output_0                                       |        /block.0.2_4/Relu_output_0        |                                               |
    |   77  |        /block.1.0_4/Conv        |        Conv       |                        /block.0.2_4/Relu_output_0, onnx::Conv_563, onnx::Conv_564                       |        /block.1.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   78  |        /block.1.2_3/Relu        |        Relu       |                                        /block.1.0_4/Conv_output_0                                       |        /block.1.2_3/Relu_output_0        |                                               |
    |   79  |   /avgpool_1/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_3/Relu_output_0                                       |  /avgpool_1/GlobalAveragePool_output_0   |                                               |
    |   80  |           /fc1_1/Conv           |        Conv       |  /avgpool_1/GlobalAveragePool_output_0, 1.features.5.block.2.fc1.weight, 1.features.5.block.2.fc1.bias  |           /fc1_1/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   81  |        /activation_1/Relu       |        Relu       |                                           /fc1_1/Conv_output_0                                          |       /activation_1/Relu_output_0        |                                               |
    |   82  |           /fc2_1/Conv           |        Conv       |       /activation_1/Relu_output_0, 1.features.5.block.2.fc2.weight, 1.features.5.block.2.fc2.bias       |           /fc2_1/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   83  | /scale_activation_1/HardSigmoid |    HardSigmoid    |                                           /fc2_1/Conv_output_0                                          | /scale_activation_1/HardSigmoid_output_0 |                     alpha                     |
    |   84  |              /Mul_1             |        Mul        |                   /scale_activation_1/HardSigmoid_output_0, /block.1.2_3/Relu_output_0                  |             /Mul_1_output_0              |                                               |
    |   85  |        /block.3.0_1/Conv        |        Conv       |                             /Mul_1_output_0, onnx::Conv_566, onnx::Conv_567                             |        /block.3.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   86  |              /Add_2             |        Add        |                           /block.3.0_1/Conv_output_0, /block.3.0/Conv_output_0                          |             /Add_2_output_0              |                                               |
    |   87  |        /block.0.0_5/Conv        |        Conv       |                             /Add_2_output_0, onnx::Conv_569, onnx::Conv_570                             |        /block.0.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   88  |        /block.0.2_5/Relu        |        Relu       |                                        /block.0.0_5/Conv_output_0                                       |        /block.0.2_5/Relu_output_0        |                                               |
    |   89  |        /block.1.0_5/Conv        |        Conv       |                        /block.0.2_5/Relu_output_0, onnx::Conv_572, onnx::Conv_573                       |        /block.1.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   90  |        /block.1.2_4/Relu        |        Relu       |                                        /block.1.0_5/Conv_output_0                                       |        /block.1.2_4/Relu_output_0        |                                               |
    |   91  |   /avgpool_2/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_4/Relu_output_0                                       |  /avgpool_2/GlobalAveragePool_output_0   |                                               |
    |   92  |           /fc1_2/Conv           |        Conv       |  /avgpool_2/GlobalAveragePool_output_0, 1.features.6.block.2.fc1.weight, 1.features.6.block.2.fc1.bias  |           /fc1_2/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   93  |        /activation_2/Relu       |        Relu       |                                           /fc1_2/Conv_output_0                                          |       /activation_2/Relu_output_0        |                                               |
    |   94  |           /fc2_2/Conv           |        Conv       |       /activation_2/Relu_output_0, 1.features.6.block.2.fc2.weight, 1.features.6.block.2.fc2.bias       |           /fc2_2/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   95  | /scale_activation_2/HardSigmoid |    HardSigmoid    |                                           /fc2_2/Conv_output_0                                          | /scale_activation_2/HardSigmoid_output_0 |                     alpha                     |
    |   96  |              /Mul_2             |        Mul        |                   /scale_activation_2/HardSigmoid_output_0, /block.1.2_4/Relu_output_0                  |             /Mul_2_output_0              |                                               |
    |   97  |        /block.3.0_2/Conv        |        Conv       |                             /Mul_2_output_0, onnx::Conv_575, onnx::Conv_576                             |        /block.3.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   98  |              /Add_3             |        Add        |                               /block.3.0_2/Conv_output_0, /Add_2_output_0                               |             /Add_3_output_0              |                                               |
    |   99  |        /block.0.0_6/Conv        |        Conv       |                             /Add_3_output_0, onnx::Conv_578, onnx::Conv_579                             |        /block.0.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  100  |     /block.0.2_6/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_6/Conv_output_0                                       |    /block.0.2_6/HardSigmoid_output_0     |                     alpha                     |
    |  101  |         /block.0.2_6/Mul        |        Mul        |                      /block.0.0_6/Conv_output_0, /block.0.2_6/HardSigmoid_output_0                      |        /block.0.2_6/Mul_output_0         |                                               |
    |  102  |        /block.1.0_6/Conv        |        Conv       |                        /block.0.2_6/Mul_output_0, onnx::Conv_581, onnx::Conv_582                        |        /block.1.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  103  |     /block.1.2_5/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_6/Conv_output_0                                       |    /block.1.2_5/HardSigmoid_output_0     |                     alpha                     |
    |  104  |         /block.1.2_5/Mul        |        Mul        |                      /block.1.0_6/Conv_output_0, /block.1.2_5/HardSigmoid_output_0                      |        /block.1.2_5/Mul_output_0         |                                               |
    |  105  |        /block.2.0_2/Conv        |        Conv       |                        /block.1.2_5/Mul_output_0, onnx::Conv_584, onnx::Conv_585                        |        /block.2.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  106  |        /block.0.0_7/Conv        |        Conv       |                        /block.2.0_2/Conv_output_0, onnx::Conv_587, onnx::Conv_588                       |        /block.0.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  107  |     /block.0.2_7/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_7/Conv_output_0                                       |    /block.0.2_7/HardSigmoid_output_0     |                     alpha                     |
    |  108  |         /block.0.2_7/Mul        |        Mul        |                      /block.0.0_7/Conv_output_0, /block.0.2_7/HardSigmoid_output_0                      |        /block.0.2_7/Mul_output_0         |                                               |
    |  109  |        /block.1.0_7/Conv        |        Conv       |                        /block.0.2_7/Mul_output_0, onnx::Conv_590, onnx::Conv_591                        |        /block.1.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  110  |     /block.1.2_6/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_7/Conv_output_0                                       |    /block.1.2_6/HardSigmoid_output_0     |                     alpha                     |
    |  111  |         /block.1.2_6/Mul        |        Mul        |                      /block.1.0_7/Conv_output_0, /block.1.2_6/HardSigmoid_output_0                      |        /block.1.2_6/Mul_output_0         |                                               |
    |  112  |        /block.2.0_3/Conv        |        Conv       |                        /block.1.2_6/Mul_output_0, onnx::Conv_593, onnx::Conv_594                        |        /block.2.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  113  |              /Add_4             |        Add        |                          /block.2.0_3/Conv_output_0, /block.2.0_2/Conv_output_0                         |             /Add_4_output_0              |                                               |
    |  114  |        /block.0.0_8/Conv        |        Conv       |                             /Add_4_output_0, onnx::Conv_596, onnx::Conv_597                             |        /block.0.0_8/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  115  |     /block.0.2_8/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_8/Conv_output_0                                       |    /block.0.2_8/HardSigmoid_output_0     |                     alpha                     |
    |  116  |         /block.0.2_8/Mul        |        Mul        |                      /block.0.0_8/Conv_output_0, /block.0.2_8/HardSigmoid_output_0                      |        /block.0.2_8/Mul_output_0         |                                               |
    |  117  |        /block.1.0_8/Conv        |        Conv       |                        /block.0.2_8/Mul_output_0, onnx::Conv_599, onnx::Conv_600                        |        /block.1.0_8/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  118  |     /block.1.2_7/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_8/Conv_output_0                                       |    /block.1.2_7/HardSigmoid_output_0     |                     alpha                     |
    |  119  |         /block.1.2_7/Mul        |        Mul        |                      /block.1.0_8/Conv_output_0, /block.1.2_7/HardSigmoid_output_0                      |        /block.1.2_7/Mul_output_0         |                                               |
    |  120  |        /block.2.0_4/Conv        |        Conv       |                        /block.1.2_7/Mul_output_0, onnx::Conv_602, onnx::Conv_603                        |        /block.2.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  121  |              /Add_5             |        Add        |                               /block.2.0_4/Conv_output_0, /Add_4_output_0                               |             /Add_5_output_0              |                                               |
    |  122  |        /block.0.0_9/Conv        |        Conv       |                             /Add_5_output_0, onnx::Conv_605, onnx::Conv_606                             |        /block.0.0_9/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  123  |     /block.0.2_9/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_9/Conv_output_0                                       |    /block.0.2_9/HardSigmoid_output_0     |                     alpha                     |
    |  124  |         /block.0.2_9/Mul        |        Mul        |                      /block.0.0_9/Conv_output_0, /block.0.2_9/HardSigmoid_output_0                      |        /block.0.2_9/Mul_output_0         |                                               |
    |  125  |        /block.1.0_9/Conv        |        Conv       |                        /block.0.2_9/Mul_output_0, onnx::Conv_608, onnx::Conv_609                        |        /block.1.0_9/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  126  |     /block.1.2_8/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_9/Conv_output_0                                       |    /block.1.2_8/HardSigmoid_output_0     |                     alpha                     |
    |  127  |         /block.1.2_8/Mul        |        Mul        |                      /block.1.0_9/Conv_output_0, /block.1.2_8/HardSigmoid_output_0                      |        /block.1.2_8/Mul_output_0         |                                               |
    |  128  |        /block.2.0_5/Conv        |        Conv       |                        /block.1.2_8/Mul_output_0, onnx::Conv_611, onnx::Conv_612                        |        /block.2.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  129  |              /Add_6             |        Add        |                               /block.2.0_5/Conv_output_0, /Add_5_output_0                               |             /Add_6_output_0              |                                               |
    |  130  |        /block.0.0_10/Conv       |        Conv       |                             /Add_6_output_0, onnx::Conv_614, onnx::Conv_615                             |       /block.0.0_10/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  131  |    /block.0.2_10/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_10/Conv_output_0                                       |    /block.0.2_10/HardSigmoid_output_0    |                     alpha                     |
    |  132  |        /block.0.2_10/Mul        |        Mul        |                     /block.0.0_10/Conv_output_0, /block.0.2_10/HardSigmoid_output_0                     |        /block.0.2_10/Mul_output_0        |                                               |
    |  133  |        /block.1.0_10/Conv       |        Conv       |                        /block.0.2_10/Mul_output_0, onnx::Conv_617, onnx::Conv_618                       |       /block.1.0_10/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  134  |     /block.1.2_9/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_10/Conv_output_0                                       |    /block.1.2_9/HardSigmoid_output_0     |                     alpha                     |
    |  135  |         /block.1.2_9/Mul        |        Mul        |                      /block.1.0_10/Conv_output_0, /block.1.2_9/HardSigmoid_output_0                     |        /block.1.2_9/Mul_output_0         |                                               |
    |  136  |   /avgpool_3/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_9/Mul_output_0                                        |  /avgpool_3/GlobalAveragePool_output_0   |                                               |
    |  137  |           /fc1_3/Conv           |        Conv       | /avgpool_3/GlobalAveragePool_output_0, 1.features.11.block.2.fc1.weight, 1.features.11.block.2.fc1.bias |           /fc1_3/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  138  |        /activation_3/Relu       |        Relu       |                                           /fc1_3/Conv_output_0                                          |       /activation_3/Relu_output_0        |                                               |
    |  139  |           /fc2_3/Conv           |        Conv       |      /activation_3/Relu_output_0, 1.features.11.block.2.fc2.weight, 1.features.11.block.2.fc2.bias      |           /fc2_3/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  140  | /scale_activation_3/HardSigmoid |    HardSigmoid    |                                           /fc2_3/Conv_output_0                                          | /scale_activation_3/HardSigmoid_output_0 |                     alpha                     |
    |  141  |              /Mul_3             |        Mul        |                   /scale_activation_3/HardSigmoid_output_0, /block.1.2_9/Mul_output_0                   |             /Mul_3_output_0              |                                               |
    |  142  |        /block.3.0_3/Conv        |        Conv       |                             /Mul_3_output_0, onnx::Conv_620, onnx::Conv_621                             |        /block.3.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  143  |        /block.0.0_11/Conv       |        Conv       |                        /block.3.0_3/Conv_output_0, onnx::Conv_623, onnx::Conv_624                       |       /block.0.0_11/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  144  |    /block.0.2_11/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_11/Conv_output_0                                       |    /block.0.2_11/HardSigmoid_output_0    |                     alpha                     |
    |  145  |        /block.0.2_11/Mul        |        Mul        |                     /block.0.0_11/Conv_output_0, /block.0.2_11/HardSigmoid_output_0                     |        /block.0.2_11/Mul_output_0        |                                               |
    |  146  |        /block.1.0_11/Conv       |        Conv       |                        /block.0.2_11/Mul_output_0, onnx::Conv_626, onnx::Conv_627                       |       /block.1.0_11/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  147  |    /block.1.2_10/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_11/Conv_output_0                                       |    /block.1.2_10/HardSigmoid_output_0    |                     alpha                     |
    |  148  |        /block.1.2_10/Mul        |        Mul        |                     /block.1.0_11/Conv_output_0, /block.1.2_10/HardSigmoid_output_0                     |        /block.1.2_10/Mul_output_0        |                                               |
    |  149  |   /avgpool_4/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_10/Mul_output_0                                       |  /avgpool_4/GlobalAveragePool_output_0   |                                               |
    |  150  |           /fc1_4/Conv           |        Conv       | /avgpool_4/GlobalAveragePool_output_0, 1.features.12.block.2.fc1.weight, 1.features.12.block.2.fc1.bias |           /fc1_4/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  151  |        /activation_4/Relu       |        Relu       |                                           /fc1_4/Conv_output_0                                          |       /activation_4/Relu_output_0        |                                               |
    |  152  |           /fc2_4/Conv           |        Conv       |      /activation_4/Relu_output_0, 1.features.12.block.2.fc2.weight, 1.features.12.block.2.fc2.bias      |           /fc2_4/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  153  | /scale_activation_4/HardSigmoid |    HardSigmoid    |                                           /fc2_4/Conv_output_0                                          | /scale_activation_4/HardSigmoid_output_0 |                     alpha                     |
    |  154  |              /Mul_4             |        Mul        |                   /scale_activation_4/HardSigmoid_output_0, /block.1.2_10/Mul_output_0                  |             /Mul_4_output_0              |                                               |
    |  155  |        /block.3.0_4/Conv        |        Conv       |                             /Mul_4_output_0, onnx::Conv_629, onnx::Conv_630                             |        /block.3.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  156  |              /Add_7             |        Add        |                          /block.3.0_4/Conv_output_0, /block.3.0_3/Conv_output_0                         |             /Add_7_output_0              |                                               |
    |  157  |        /block.0.0_12/Conv       |        Conv       |                             /Add_7_output_0, onnx::Conv_632, onnx::Conv_633                             |       /block.0.0_12/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  158  |    /block.0.2_12/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_12/Conv_output_0                                       |    /block.0.2_12/HardSigmoid_output_0    |                     alpha                     |
    |  159  |        /block.0.2_12/Mul        |        Mul        |                     /block.0.0_12/Conv_output_0, /block.0.2_12/HardSigmoid_output_0                     |        /block.0.2_12/Mul_output_0        |                                               |
    |  160  |        /block.1.0_12/Conv       |        Conv       |                        /block.0.2_12/Mul_output_0, onnx::Conv_635, onnx::Conv_636                       |       /block.1.0_12/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  161  |    /block.1.2_11/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_12/Conv_output_0                                       |    /block.1.2_11/HardSigmoid_output_0    |                     alpha                     |
    |  162  |        /block.1.2_11/Mul        |        Mul        |                     /block.1.0_12/Conv_output_0, /block.1.2_11/HardSigmoid_output_0                     |        /block.1.2_11/Mul_output_0        |                                               |
    |  163  |   /avgpool_5/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_11/Mul_output_0                                       |  /avgpool_5/GlobalAveragePool_output_0   |                                               |
    |  164  |           /fc1_5/Conv           |        Conv       | /avgpool_5/GlobalAveragePool_output_0, 1.features.13.block.2.fc1.weight, 1.features.13.block.2.fc1.bias |           /fc1_5/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  165  |        /activation_5/Relu       |        Relu       |                                           /fc1_5/Conv_output_0                                          |       /activation_5/Relu_output_0        |                                               |
    |  166  |           /fc2_5/Conv           |        Conv       |      /activation_5/Relu_output_0, 1.features.13.block.2.fc2.weight, 1.features.13.block.2.fc2.bias      |           /fc2_5/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  167  | /scale_activation_5/HardSigmoid |    HardSigmoid    |                                           /fc2_5/Conv_output_0                                          | /scale_activation_5/HardSigmoid_output_0 |                     alpha                     |
    |  168  |              /Mul_5             |        Mul        |                   /scale_activation_5/HardSigmoid_output_0, /block.1.2_11/Mul_output_0                  |             /Mul_5_output_0              |                                               |
    |  169  |        /block.3.0_5/Conv        |        Conv       |                             /Mul_5_output_0, onnx::Conv_638, onnx::Conv_639                             |        /block.3.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  170  |        /block.0.0_13/Conv       |        Conv       |                        /block.3.0_5/Conv_output_0, onnx::Conv_641, onnx::Conv_642                       |       /block.0.0_13/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  171  |    /block.0.2_13/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_13/Conv_output_0                                       |    /block.0.2_13/HardSigmoid_output_0    |                     alpha                     |
    |  172  |        /block.0.2_13/Mul        |        Mul        |                     /block.0.0_13/Conv_output_0, /block.0.2_13/HardSigmoid_output_0                     |        /block.0.2_13/Mul_output_0        |                                               |
    |  173  |        /block.1.0_13/Conv       |        Conv       |                        /block.0.2_13/Mul_output_0, onnx::Conv_644, onnx::Conv_645                       |       /block.1.0_13/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  174  |    /block.1.2_12/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_13/Conv_output_0                                       |    /block.1.2_12/HardSigmoid_output_0    |                     alpha                     |
    |  175  |        /block.1.2_12/Mul        |        Mul        |                     /block.1.0_13/Conv_output_0, /block.1.2_12/HardSigmoid_output_0                     |        /block.1.2_12/Mul_output_0        |                                               |
    |  176  |   /avgpool_6/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_12/Mul_output_0                                       |  /avgpool_6/GlobalAveragePool_output_0   |                                               |
    |  177  |           /fc1_6/Conv           |        Conv       | /avgpool_6/GlobalAveragePool_output_0, 1.features.14.block.2.fc1.weight, 1.features.14.block.2.fc1.bias |           /fc1_6/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  178  |        /activation_6/Relu       |        Relu       |                                           /fc1_6/Conv_output_0                                          |       /activation_6/Relu_output_0        |                                               |
    |  179  |           /fc2_6/Conv           |        Conv       |      /activation_6/Relu_output_0, 1.features.14.block.2.fc2.weight, 1.features.14.block.2.fc2.bias      |           /fc2_6/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  180  | /scale_activation_6/HardSigmoid |    HardSigmoid    |                                           /fc2_6/Conv_output_0                                          | /scale_activation_6/HardSigmoid_output_0 |                     alpha                     |
    |  181  |              /Mul_6             |        Mul        |                   /scale_activation_6/HardSigmoid_output_0, /block.1.2_12/Mul_output_0                  |             /Mul_6_output_0              |                                               |
    |  182  |        /block.3.0_6/Conv        |        Conv       |                             /Mul_6_output_0, onnx::Conv_647, onnx::Conv_648                             |        /block.3.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  183  |              /Add_8             |        Add        |                          /block.3.0_6/Conv_output_0, /block.3.0_5/Conv_output_0                         |             /Add_8_output_0              |                                               |
    |  184  |        /block.0.0_14/Conv       |        Conv       |                             /Add_8_output_0, onnx::Conv_650, onnx::Conv_651                             |       /block.0.0_14/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  185  |    /block.0.2_14/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_14/Conv_output_0                                       |    /block.0.2_14/HardSigmoid_output_0    |                     alpha                     |
    |  186  |        /block.0.2_14/Mul        |        Mul        |                     /block.0.0_14/Conv_output_0, /block.0.2_14/HardSigmoid_output_0                     |        /block.0.2_14/Mul_output_0        |                                               |
    |  187  |        /block.1.0_14/Conv       |        Conv       |                        /block.0.2_14/Mul_output_0, onnx::Conv_653, onnx::Conv_654                       |       /block.1.0_14/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  188  |    /block.1.2_13/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_14/Conv_output_0                                       |    /block.1.2_13/HardSigmoid_output_0    |                     alpha                     |
    |  189  |        /block.1.2_13/Mul        |        Mul        |                     /block.1.0_14/Conv_output_0, /block.1.2_13/HardSigmoid_output_0                     |        /block.1.2_13/Mul_output_0        |                                               |
    |  190  |   /avgpool_7/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_13/Mul_output_0                                       |  /avgpool_7/GlobalAveragePool_output_0   |                                               |
    |  191  |           /fc1_7/Conv           |        Conv       | /avgpool_7/GlobalAveragePool_output_0, 1.features.15.block.2.fc1.weight, 1.features.15.block.2.fc1.bias |           /fc1_7/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  192  |        /activation_7/Relu       |        Relu       |                                           /fc1_7/Conv_output_0                                          |       /activation_7/Relu_output_0        |                                               |
    |  193  |           /fc2_7/Conv           |        Conv       |      /activation_7/Relu_output_0, 1.features.15.block.2.fc2.weight, 1.features.15.block.2.fc2.bias      |           /fc2_7/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  194  | /scale_activation_7/HardSigmoid |    HardSigmoid    |                                           /fc2_7/Conv_output_0                                          | /scale_activation_7/HardSigmoid_output_0 |                     alpha                     |
    |  195  |              /Mul_7             |        Mul        |                   /scale_activation_7/HardSigmoid_output_0, /block.1.2_13/Mul_output_0                  |             /Mul_7_output_0              |                                               |
    |  196  |        /block.3.0_7/Conv        |        Conv       |                             /Mul_7_output_0, onnx::Conv_656, onnx::Conv_657                             |        /block.3.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  197  |              /Add_9             |        Add        |                               /block.3.0_7/Conv_output_0, /Add_8_output_0                               |             /Add_9_output_0              |                                               |
    |  198  |       /features.16.0/Conv       |        Conv       |                             /Add_9_output_0, onnx::Conv_659, onnx::Conv_660                             |       /features.16.0/Conv_output_0       | dilations, group, kernel_shape, pads, strides |
    |  199  |    /features.16.2/HardSigmoid   |    HardSigmoid    |                                       /features.16.0/Conv_output_0                                      |   /features.16.2/HardSigmoid_output_0    |                     alpha                     |
    |  200  |        /features.16.2/Mul       |        Mul        |                    /features.16.0/Conv_output_0, /features.16.2/HardSigmoid_output_0                    |       /features.16.2/Mul_output_0        |                                               |
    |  201  |   /avgpool_8/GlobalAveragePool  | GlobalAveragePool |                                       /features.16.2/Mul_output_0                                       |  /avgpool_8/GlobalAveragePool_output_0   |                                               |
    |  202  |             /Flatten            |      Flatten      |                                  /avgpool_8/GlobalAveragePool_output_0                                  |            /Flatten_output_0             |                      axis                     |
    |  203  |        /classifier.0/Gemm       |        Gemm       |                      /Flatten_output_0, 1.classifier.0.weight, 1.classifier.0.bias                      |       /classifier.0/Gemm_output_0        |              alpha, beta, transB              |
    |  204  |    /classifier.1/HardSigmoid    |    HardSigmoid    |                                       /classifier.0/Gemm_output_0                                       |    /classifier.1/HardSigmoid_output_0    |                     alpha                     |
    |  205  |        /classifier.1/Mul        |        Mul        |                     /classifier.0/Gemm_output_0, /classifier.1/HardSigmoid_output_0                     |        /classifier.1/Mul_output_0        |                                               |
    |  206  |        /classifier.3/Gemm       |        Gemm       |                  /classifier.1/Mul_output_0, 1.classifier.3.weight, 1.classifier.3.bias                 |                   522                    |              alpha, beta, transB              |
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+[0m
    I0327 21:02:33.118024 139809641850688 onnx_runtime.py:90] ONNX Model Summary: 
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+
    | Index |               Name              |        Type       |                                                  Inputs                                                 |                 Outputs                  |                   Attributes                  |
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+
    |   0   |            Identity_0           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_660              |                                               |
    |   1   |            Identity_1           |      Identity     |                                              onnx::Conv_639                                             |              onnx::Conv_657              |                                               |
    |   2   |            Identity_2           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_654              |                                               |
    |   3   |            Identity_3           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_651              |                                               |
    |   4   |            Identity_4           |      Identity     |                                              onnx::Conv_639                                             |              onnx::Conv_648              |                                               |
    |   5   |            Identity_5           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_645              |                                               |
    |   6   |            Identity_6           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_642              |                                               |
    |   7   |            Identity_7           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_636              |                                               |
    |   8   |            Identity_8           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_633              |                                               |
    |   9   |            Identity_9           |      Identity     |                                              onnx::Conv_621                                             |              onnx::Conv_630              |                                               |
    |   10  |           Identity_10           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_627              |                                               |
    |   11  |           Identity_11           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_624              |                                               |
    |   12  |           Identity_12           |      Identity     |                                      1.features.11.block.2.fc2.bias                                     |              onnx::Conv_618              |                                               |
    |   13  |           Identity_13           |      Identity     |                                      1.features.11.block.2.fc2.bias                                     |              onnx::Conv_615              |                                               |
    |   14  |           Identity_14           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_612              |                                               |
    |   15  |           Identity_15           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_609              |                                               |
    |   16  |           Identity_16           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_606              |                                               |
    |   17  |           Identity_17           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_603              |                                               |
    |   18  |           Identity_18           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_600              |                                               |
    |   19  |           Identity_19           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_594              |                                               |
    |   20  |           Identity_20           |      Identity     |                                              onnx::Conv_588                                             |              onnx::Conv_591              |                                               |
    |   21  |           Identity_21           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |              onnx::Conv_582              |                                               |
    |   22  |           Identity_22           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |              onnx::Conv_579              |                                               |
    |   23  |           Identity_23           |      Identity     |                                              onnx::Conv_558                                             |              onnx::Conv_576              |                                               |
    |   24  |           Identity_24           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_573              |                                               |
    |   25  |           Identity_25           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_570              |                                               |
    |   26  |           Identity_26           |      Identity     |                                              onnx::Conv_558                                             |              onnx::Conv_567              |                                               |
    |   27  |           Identity_27           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_564              |                                               |
    |   28  |           Identity_28           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_561              |                                               |
    |   29  |           Identity_29           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_555              |                                               |
    |   30  |           Identity_30           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_552              |                                               |
    |   31  |           Identity_31           |      Identity     |                                      1.features.4.block.2.fc1.bias                                      |              onnx::Conv_549              |                                               |
    |   32  |           Identity_32           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_546              |                                               |
    |   33  |           Identity_33           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_543              |                                               |
    |   34  |           Identity_34           |      Identity     |                                      1.features.4.block.2.fc1.bias                                      |              onnx::Conv_540              |                                               |
    |   35  |           Identity_35           |      Identity     |                                              onnx::Conv_534                                             |              onnx::Conv_537              |                                               |
    |   36  |           Identity_36           |      Identity     |                                              onnx::Conv_525                                             |              onnx::Conv_531              |                                               |
    |   37  |           Identity_37           |      Identity     |                                              onnx::Conv_525                                             |              onnx::Conv_528              |                                               |
    |   38  |           Identity_38           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |      1.features.15.block.2.fc2.bias      |                                               |
    |   39  |           Identity_39           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |      1.features.15.block.2.fc1.bias      |                                               |
    |   40  |           Identity_40           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |      1.features.13.block.2.fc2.bias      |                                               |
    |   41  |           Identity_41           |      Identity     |                                      1.features.12.block.2.fc1.bias                                     |      1.features.13.block.2.fc1.bias      |                                               |
    |   42  |           Identity_42           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |      1.features.11.block.2.fc1.bias      |                                               |
    |   43  |           Identity_43           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |      1.features.6.block.2.fc2.bias       |                                               |
    |   44  |           Identity_44           |      Identity     |                                      1.features.5.block.2.fc1.bias                                      |      1.features.6.block.2.fc1.bias       |                                               |
    |   45  |             /0/Conv             |        Conv       |                                         input, 0.weight, 0.bias                                         |             /0/Conv_output_0             | dilations, group, kernel_shape, pads, strides |
    |   46  |        /features.0.0/Conv       |        Conv       |                             /0/Conv_output_0, onnx::Conv_524, onnx::Conv_525                            |       /features.0.0/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   47  |    /features.0.2/HardSigmoid    |    HardSigmoid    |                                       /features.0.0/Conv_output_0                                       |    /features.0.2/HardSigmoid_output_0    |                     alpha                     |
    |   48  |        /features.0.2/Mul        |        Mul        |                     /features.0.0/Conv_output_0, /features.0.2/HardSigmoid_output_0                     |        /features.0.2/Mul_output_0        |                                               |
    |   49  |         /block.0.0/Conv         |        Conv       |                        /features.0.2/Mul_output_0, onnx::Conv_527, onnx::Conv_528                       |         /block.0.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   50  |         /block.0.2/Relu         |        Relu       |                                         /block.0.0/Conv_output_0                                        |         /block.0.2/Relu_output_0         |                                               |
    |   51  |         /block.1.0/Conv         |        Conv       |                         /block.0.2/Relu_output_0, onnx::Conv_530, onnx::Conv_531                        |         /block.1.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   52  |               /Add              |        Add        |                           /block.1.0/Conv_output_0, /features.0.2/Mul_output_0                          |              /Add_output_0               |                                               |
    |   53  |        /block.0.0_1/Conv        |        Conv       |                              /Add_output_0, onnx::Conv_533, onnx::Conv_534                              |        /block.0.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   54  |        /block.0.2_1/Relu        |        Relu       |                                        /block.0.0_1/Conv_output_0                                       |        /block.0.2_1/Relu_output_0        |                                               |
    |   55  |        /block.1.0_1/Conv        |        Conv       |                        /block.0.2_1/Relu_output_0, onnx::Conv_536, onnx::Conv_537                       |        /block.1.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   56  |         /block.1.2/Relu         |        Relu       |                                        /block.1.0_1/Conv_output_0                                       |         /block.1.2/Relu_output_0         |                                               |
    |   57  |         /block.2.0/Conv         |        Conv       |                         /block.1.2/Relu_output_0, onnx::Conv_539, onnx::Conv_540                        |         /block.2.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   58  |        /block.0.0_2/Conv        |        Conv       |                         /block.2.0/Conv_output_0, onnx::Conv_542, onnx::Conv_543                        |        /block.0.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   59  |        /block.0.2_2/Relu        |        Relu       |                                        /block.0.0_2/Conv_output_0                                       |        /block.0.2_2/Relu_output_0        |                                               |
    |   60  |        /block.1.0_2/Conv        |        Conv       |                        /block.0.2_2/Relu_output_0, onnx::Conv_545, onnx::Conv_546                       |        /block.1.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   61  |        /block.1.2_1/Relu        |        Relu       |                                        /block.1.0_2/Conv_output_0                                       |        /block.1.2_1/Relu_output_0        |                                               |
    |   62  |        /block.2.0_1/Conv        |        Conv       |                        /block.1.2_1/Relu_output_0, onnx::Conv_548, onnx::Conv_549                       |        /block.2.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   63  |              /Add_1             |        Add        |                           /block.2.0_1/Conv_output_0, /block.2.0/Conv_output_0                          |             /Add_1_output_0              |                                               |
    |   64  |        /block.0.0_3/Conv        |        Conv       |                             /Add_1_output_0, onnx::Conv_551, onnx::Conv_552                             |        /block.0.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   65  |        /block.0.2_3/Relu        |        Relu       |                                        /block.0.0_3/Conv_output_0                                       |        /block.0.2_3/Relu_output_0        |                                               |
    |   66  |        /block.1.0_3/Conv        |        Conv       |                        /block.0.2_3/Relu_output_0, onnx::Conv_554, onnx::Conv_555                       |        /block.1.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   67  |        /block.1.2_2/Relu        |        Relu       |                                        /block.1.0_3/Conv_output_0                                       |        /block.1.2_2/Relu_output_0        |                                               |
    |   68  |    /avgpool/GlobalAveragePool   | GlobalAveragePool |                                        /block.1.2_2/Relu_output_0                                       |   /avgpool/GlobalAveragePool_output_0    |                                               |
    |   69  |            /fc1/Conv            |        Conv       |   /avgpool/GlobalAveragePool_output_0, 1.features.4.block.2.fc1.weight, 1.features.4.block.2.fc1.bias   |            /fc1/Conv_output_0            | dilations, group, kernel_shape, pads, strides |
    |   70  |         /activation/Relu        |        Relu       |                                            /fc1/Conv_output_0                                           |        /activation/Relu_output_0         |                                               |
    |   71  |            /fc2/Conv            |        Conv       |        /activation/Relu_output_0, 1.features.4.block.2.fc2.weight, 1.features.4.block.2.fc2.bias        |            /fc2/Conv_output_0            | dilations, group, kernel_shape, pads, strides |
    |   72  |  /scale_activation/HardSigmoid  |    HardSigmoid    |                                            /fc2/Conv_output_0                                           |  /scale_activation/HardSigmoid_output_0  |                     alpha                     |
    |   73  |               /Mul              |        Mul        |                    /scale_activation/HardSigmoid_output_0, /block.1.2_2/Relu_output_0                   |              /Mul_output_0               |                                               |
    |   74  |         /block.3.0/Conv         |        Conv       |                              /Mul_output_0, onnx::Conv_557, onnx::Conv_558                              |         /block.3.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   75  |        /block.0.0_4/Conv        |        Conv       |                         /block.3.0/Conv_output_0, onnx::Conv_560, onnx::Conv_561                        |        /block.0.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   76  |        /block.0.2_4/Relu        |        Relu       |                                        /block.0.0_4/Conv_output_0                                       |        /block.0.2_4/Relu_output_0        |                                               |
    |   77  |        /block.1.0_4/Conv        |        Conv       |                        /block.0.2_4/Relu_output_0, onnx::Conv_563, onnx::Conv_564                       |        /block.1.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   78  |        /block.1.2_3/Relu        |        Relu       |                                        /block.1.0_4/Conv_output_0                                       |        /block.1.2_3/Relu_output_0        |                                               |
    |   79  |   /avgpool_1/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_3/Relu_output_0                                       |  /avgpool_1/GlobalAveragePool_output_0   |                                               |
    |   80  |           /fc1_1/Conv           |        Conv       |  /avgpool_1/GlobalAveragePool_output_0, 1.features.5.block.2.fc1.weight, 1.features.5.block.2.fc1.bias  |           /fc1_1/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   81  |        /activation_1/Relu       |        Relu       |                                           /fc1_1/Conv_output_0                                          |       /activation_1/Relu_output_0        |                                               |
    |   82  |           /fc2_1/Conv           |        Conv       |       /activation_1/Relu_output_0, 1.features.5.block.2.fc2.weight, 1.features.5.block.2.fc2.bias       |           /fc2_1/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   83  | /scale_activation_1/HardSigmoid |    HardSigmoid    |                                           /fc2_1/Conv_output_0                                          | /scale_activation_1/HardSigmoid_output_0 |                     alpha                     |
    |   84  |              /Mul_1             |        Mul        |                   /scale_activation_1/HardSigmoid_output_0, /block.1.2_3/Relu_output_0                  |             /Mul_1_output_0              |                                               |
    |   85  |        /block.3.0_1/Conv        |        Conv       |                             /Mul_1_output_0, onnx::Conv_566, onnx::Conv_567                             |        /block.3.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   86  |              /Add_2             |        Add        |                           /block.3.0_1/Conv_output_0, /block.3.0/Conv_output_0                          |             /Add_2_output_0              |                                               |
    |   87  |        /block.0.0_5/Conv        |        Conv       |                             /Add_2_output_0, onnx::Conv_569, onnx::Conv_570                             |        /block.0.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   88  |        /block.0.2_5/Relu        |        Relu       |                                        /block.0.0_5/Conv_output_0                                       |        /block.0.2_5/Relu_output_0        |                                               |
    |   89  |        /block.1.0_5/Conv        |        Conv       |                        /block.0.2_5/Relu_output_0, onnx::Conv_572, onnx::Conv_573                       |        /block.1.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   90  |        /block.1.2_4/Relu        |        Relu       |                                        /block.1.0_5/Conv_output_0                                       |        /block.1.2_4/Relu_output_0        |                                               |
    |   91  |   /avgpool_2/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_4/Relu_output_0                                       |  /avgpool_2/GlobalAveragePool_output_0   |                                               |
    |   92  |           /fc1_2/Conv           |        Conv       |  /avgpool_2/GlobalAveragePool_output_0, 1.features.6.block.2.fc1.weight, 1.features.6.block.2.fc1.bias  |           /fc1_2/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   93  |        /activation_2/Relu       |        Relu       |                                           /fc1_2/Conv_output_0                                          |       /activation_2/Relu_output_0        |                                               |
    |   94  |           /fc2_2/Conv           |        Conv       |       /activation_2/Relu_output_0, 1.features.6.block.2.fc2.weight, 1.features.6.block.2.fc2.bias       |           /fc2_2/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   95  | /scale_activation_2/HardSigmoid |    HardSigmoid    |                                           /fc2_2/Conv_output_0                                          | /scale_activation_2/HardSigmoid_output_0 |                     alpha                     |
    |   96  |              /Mul_2             |        Mul        |                   /scale_activation_2/HardSigmoid_output_0, /block.1.2_4/Relu_output_0                  |             /Mul_2_output_0              |                                               |
    |   97  |        /block.3.0_2/Conv        |        Conv       |                             /Mul_2_output_0, onnx::Conv_575, onnx::Conv_576                             |        /block.3.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   98  |              /Add_3             |        Add        |                               /block.3.0_2/Conv_output_0, /Add_2_output_0                               |             /Add_3_output_0              |                                               |
    |   99  |        /block.0.0_6/Conv        |        Conv       |                             /Add_3_output_0, onnx::Conv_578, onnx::Conv_579                             |        /block.0.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  100  |     /block.0.2_6/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_6/Conv_output_0                                       |    /block.0.2_6/HardSigmoid_output_0     |                     alpha                     |
    |  101  |         /block.0.2_6/Mul        |        Mul        |                      /block.0.0_6/Conv_output_0, /block.0.2_6/HardSigmoid_output_0                      |        /block.0.2_6/Mul_output_0         |                                               |
    |  102  |        /block.1.0_6/Conv        |        Conv       |                        /block.0.2_6/Mul_output_0, onnx::Conv_581, onnx::Conv_582                        |        /block.1.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  103  |     /block.1.2_5/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_6/Conv_output_0                                       |    /block.1.2_5/HardSigmoid_output_0     |                     alpha                     |
    |  104  |         /block.1.2_5/Mul        |        Mul        |                      /block.1.0_6/Conv_output_0, /block.1.2_5/HardSigmoid_output_0                      |        /block.1.2_5/Mul_output_0         |                                               |
    |  105  |        /block.2.0_2/Conv        |        Conv       |                        /block.1.2_5/Mul_output_0, onnx::Conv_584, onnx::Conv_585                        |        /block.2.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  106  |        /block.0.0_7/Conv        |        Conv       |                        /block.2.0_2/Conv_output_0, onnx::Conv_587, onnx::Conv_588                       |        /block.0.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  107  |     /block.0.2_7/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_7/Conv_output_0                                       |    /block.0.2_7/HardSigmoid_output_0     |                     alpha                     |
    |  108  |         /block.0.2_7/Mul        |        Mul        |                      /block.0.0_7/Conv_output_0, /block.0.2_7/HardSigmoid_output_0                      |        /block.0.2_7/Mul_output_0         |                                               |
    |  109  |        /block.1.0_7/Conv        |        Conv       |                        /block.0.2_7/Mul_output_0, onnx::Conv_590, onnx::Conv_591                        |        /block.1.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  110  |     /block.1.2_6/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_7/Conv_output_0                                       |    /block.1.2_6/HardSigmoid_output_0     |                     alpha                     |
    |  111  |         /block.1.2_6/Mul        |        Mul        |                      /block.1.0_7/Conv_output_0, /block.1.2_6/HardSigmoid_output_0                      |        /block.1.2_6/Mul_output_0         |                                               |
    |  112  |        /block.2.0_3/Conv        |        Conv       |                        /block.1.2_6/Mul_output_0, onnx::Conv_593, onnx::Conv_594                        |        /block.2.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  113  |              /Add_4             |        Add        |                          /block.2.0_3/Conv_output_0, /block.2.0_2/Conv_output_0                         |             /Add_4_output_0              |                                               |
    |  114  |        /block.0.0_8/Conv        |        Conv       |                             /Add_4_output_0, onnx::Conv_596, onnx::Conv_597                             |        /block.0.0_8/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  115  |     /block.0.2_8/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_8/Conv_output_0                                       |    /block.0.2_8/HardSigmoid_output_0     |                     alpha                     |
    |  116  |         /block.0.2_8/Mul        |        Mul        |                      /block.0.0_8/Conv_output_0, /block.0.2_8/HardSigmoid_output_0                      |        /block.0.2_8/Mul_output_0         |                                               |
    |  117  |        /block.1.0_8/Conv        |        Conv       |                        /block.0.2_8/Mul_output_0, onnx::Conv_599, onnx::Conv_600                        |        /block.1.0_8/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  118  |     /block.1.2_7/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_8/Conv_output_0                                       |    /block.1.2_7/HardSigmoid_output_0     |                     alpha                     |
    |  119  |         /block.1.2_7/Mul        |        Mul        |                      /block.1.0_8/Conv_output_0, /block.1.2_7/HardSigmoid_output_0                      |        /block.1.2_7/Mul_output_0         |                                               |
    |  120  |        /block.2.0_4/Conv        |        Conv       |                        /block.1.2_7/Mul_output_0, onnx::Conv_602, onnx::Conv_603                        |        /block.2.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  121  |              /Add_5             |        Add        |                               /block.2.0_4/Conv_output_0, /Add_4_output_0                               |             /Add_5_output_0              |                                               |
    |  122  |        /block.0.0_9/Conv        |        Conv       |                             /Add_5_output_0, onnx::Conv_605, onnx::Conv_606                             |        /block.0.0_9/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  123  |     /block.0.2_9/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_9/Conv_output_0                                       |    /block.0.2_9/HardSigmoid_output_0     |                     alpha                     |
    |  124  |         /block.0.2_9/Mul        |        Mul        |                      /block.0.0_9/Conv_output_0, /block.0.2_9/HardSigmoid_output_0                      |        /block.0.2_9/Mul_output_0         |                                               |
    |  125  |        /block.1.0_9/Conv        |        Conv       |                        /block.0.2_9/Mul_output_0, onnx::Conv_608, onnx::Conv_609                        |        /block.1.0_9/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  126  |     /block.1.2_8/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_9/Conv_output_0                                       |    /block.1.2_8/HardSigmoid_output_0     |                     alpha                     |
    |  127  |         /block.1.2_8/Mul        |        Mul        |                      /block.1.0_9/Conv_output_0, /block.1.2_8/HardSigmoid_output_0                      |        /block.1.2_8/Mul_output_0         |                                               |
    |  128  |        /block.2.0_5/Conv        |        Conv       |                        /block.1.2_8/Mul_output_0, onnx::Conv_611, onnx::Conv_612                        |        /block.2.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  129  |              /Add_6             |        Add        |                               /block.2.0_5/Conv_output_0, /Add_5_output_0                               |             /Add_6_output_0              |                                               |
    |  130  |        /block.0.0_10/Conv       |        Conv       |                             /Add_6_output_0, onnx::Conv_614, onnx::Conv_615                             |       /block.0.0_10/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  131  |    /block.0.2_10/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_10/Conv_output_0                                       |    /block.0.2_10/HardSigmoid_output_0    |                     alpha                     |
    |  132  |        /block.0.2_10/Mul        |        Mul        |                     /block.0.0_10/Conv_output_0, /block.0.2_10/HardSigmoid_output_0                     |        /block.0.2_10/Mul_output_0        |                                               |
    |  133  |        /block.1.0_10/Conv       |        Conv       |                        /block.0.2_10/Mul_output_0, onnx::Conv_617, onnx::Conv_618                       |       /block.1.0_10/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  134  |     /block.1.2_9/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_10/Conv_output_0                                       |    /block.1.2_9/HardSigmoid_output_0     |                     alpha                     |
    |  135  |         /block.1.2_9/Mul        |        Mul        |                      /block.1.0_10/Conv_output_0, /block.1.2_9/HardSigmoid_output_0                     |        /block.1.2_9/Mul_output_0         |                                               |
    |  136  |   /avgpool_3/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_9/Mul_output_0                                        |  /avgpool_3/GlobalAveragePool_output_0   |                                               |
    |  137  |           /fc1_3/Conv           |        Conv       | /avgpool_3/GlobalAveragePool_output_0, 1.features.11.block.2.fc1.weight, 1.features.11.block.2.fc1.bias |           /fc1_3/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  138  |        /activation_3/Relu       |        Relu       |                                           /fc1_3/Conv_output_0                                          |       /activation_3/Relu_output_0        |                                               |
    |  139  |           /fc2_3/Conv           |        Conv       |      /activation_3/Relu_output_0, 1.features.11.block.2.fc2.weight, 1.features.11.block.2.fc2.bias      |           /fc2_3/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  140  | /scale_activation_3/HardSigmoid |    HardSigmoid    |                                           /fc2_3/Conv_output_0                                          | /scale_activation_3/HardSigmoid_output_0 |                     alpha                     |
    |  141  |              /Mul_3             |        Mul        |                   /scale_activation_3/HardSigmoid_output_0, /block.1.2_9/Mul_output_0                   |             /Mul_3_output_0              |                                               |
    |  142  |        /block.3.0_3/Conv        |        Conv       |                             /Mul_3_output_0, onnx::Conv_620, onnx::Conv_621                             |        /block.3.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  143  |        /block.0.0_11/Conv       |        Conv       |                        /block.3.0_3/Conv_output_0, onnx::Conv_623, onnx::Conv_624                       |       /block.0.0_11/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  144  |    /block.0.2_11/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_11/Conv_output_0                                       |    /block.0.2_11/HardSigmoid_output_0    |                     alpha                     |
    |  145  |        /block.0.2_11/Mul        |        Mul        |                     /block.0.0_11/Conv_output_0, /block.0.2_11/HardSigmoid_output_0                     |        /block.0.2_11/Mul_output_0        |                                               |
    |  146  |        /block.1.0_11/Conv       |        Conv       |                        /block.0.2_11/Mul_output_0, onnx::Conv_626, onnx::Conv_627                       |       /block.1.0_11/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  147  |    /block.1.2_10/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_11/Conv_output_0                                       |    /block.1.2_10/HardSigmoid_output_0    |                     alpha                     |
    |  148  |        /block.1.2_10/Mul        |        Mul        |                     /block.1.0_11/Conv_output_0, /block.1.2_10/HardSigmoid_output_0                     |        /block.1.2_10/Mul_output_0        |                                               |
    |  149  |   /avgpool_4/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_10/Mul_output_0                                       |  /avgpool_4/GlobalAveragePool_output_0   |                                               |
    |  150  |           /fc1_4/Conv           |        Conv       | /avgpool_4/GlobalAveragePool_output_0, 1.features.12.block.2.fc1.weight, 1.features.12.block.2.fc1.bias |           /fc1_4/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  151  |        /activation_4/Relu       |        Relu       |                                           /fc1_4/Conv_output_0                                          |       /activation_4/Relu_output_0        |                                               |
    |  152  |           /fc2_4/Conv           |        Conv       |      /activation_4/Relu_output_0, 1.features.12.block.2.fc2.weight, 1.features.12.block.2.fc2.bias      |           /fc2_4/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  153  | /scale_activation_4/HardSigmoid |    HardSigmoid    |                                           /fc2_4/Conv_output_0                                          | /scale_activation_4/HardSigmoid_output_0 |                     alpha                     |
    |  154  |              /Mul_4             |        Mul        |                   /scale_activation_4/HardSigmoid_output_0, /block.1.2_10/Mul_output_0                  |             /Mul_4_output_0              |                                               |
    |  155  |        /block.3.0_4/Conv        |        Conv       |                             /Mul_4_output_0, onnx::Conv_629, onnx::Conv_630                             |        /block.3.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  156  |              /Add_7             |        Add        |                          /block.3.0_4/Conv_output_0, /block.3.0_3/Conv_output_0                         |             /Add_7_output_0              |                                               |
    |  157  |        /block.0.0_12/Conv       |        Conv       |                             /Add_7_output_0, onnx::Conv_632, onnx::Conv_633                             |       /block.0.0_12/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  158  |    /block.0.2_12/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_12/Conv_output_0                                       |    /block.0.2_12/HardSigmoid_output_0    |                     alpha                     |
    |  159  |        /block.0.2_12/Mul        |        Mul        |                     /block.0.0_12/Conv_output_0, /block.0.2_12/HardSigmoid_output_0                     |        /block.0.2_12/Mul_output_0        |                                               |
    |  160  |        /block.1.0_12/Conv       |        Conv       |                        /block.0.2_12/Mul_output_0, onnx::Conv_635, onnx::Conv_636                       |       /block.1.0_12/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  161  |    /block.1.2_11/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_12/Conv_output_0                                       |    /block.1.2_11/HardSigmoid_output_0    |                     alpha                     |
    |  162  |        /block.1.2_11/Mul        |        Mul        |                     /block.1.0_12/Conv_output_0, /block.1.2_11/HardSigmoid_output_0                     |        /block.1.2_11/Mul_output_0        |                                               |
    |  163  |   /avgpool_5/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_11/Mul_output_0                                       |  /avgpool_5/GlobalAveragePool_output_0   |                                               |
    |  164  |           /fc1_5/Conv           |        Conv       | /avgpool_5/GlobalAveragePool_output_0, 1.features.13.block.2.fc1.weight, 1.features.13.block.2.fc1.bias |           /fc1_5/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  165  |        /activation_5/Relu       |        Relu       |                                           /fc1_5/Conv_output_0                                          |       /activation_5/Relu_output_0        |                                               |
    |  166  |           /fc2_5/Conv           |        Conv       |      /activation_5/Relu_output_0, 1.features.13.block.2.fc2.weight, 1.features.13.block.2.fc2.bias      |           /fc2_5/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  167  | /scale_activation_5/HardSigmoid |    HardSigmoid    |                                           /fc2_5/Conv_output_0                                          | /scale_activation_5/HardSigmoid_output_0 |                     alpha                     |
    |  168  |              /Mul_5             |        Mul        |                   /scale_activation_5/HardSigmoid_output_0, /block.1.2_11/Mul_output_0                  |             /Mul_5_output_0              |                                               |
    |  169  |        /block.3.0_5/Conv        |        Conv       |                             /Mul_5_output_0, onnx::Conv_638, onnx::Conv_639                             |        /block.3.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  170  |        /block.0.0_13/Conv       |        Conv       |                        /block.3.0_5/Conv_output_0, onnx::Conv_641, onnx::Conv_642                       |       /block.0.0_13/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  171  |    /block.0.2_13/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_13/Conv_output_0                                       |    /block.0.2_13/HardSigmoid_output_0    |                     alpha                     |
    |  172  |        /block.0.2_13/Mul        |        Mul        |                     /block.0.0_13/Conv_output_0, /block.0.2_13/HardSigmoid_output_0                     |        /block.0.2_13/Mul_output_0        |                                               |
    |  173  |        /block.1.0_13/Conv       |        Conv       |                        /block.0.2_13/Mul_output_0, onnx::Conv_644, onnx::Conv_645                       |       /block.1.0_13/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  174  |    /block.1.2_12/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_13/Conv_output_0                                       |    /block.1.2_12/HardSigmoid_output_0    |                     alpha                     |
    |  175  |        /block.1.2_12/Mul        |        Mul        |                     /block.1.0_13/Conv_output_0, /block.1.2_12/HardSigmoid_output_0                     |        /block.1.2_12/Mul_output_0        |                                               |
    |  176  |   /avgpool_6/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_12/Mul_output_0                                       |  /avgpool_6/GlobalAveragePool_output_0   |                                               |
    |  177  |           /fc1_6/Conv           |        Conv       | /avgpool_6/GlobalAveragePool_output_0, 1.features.14.block.2.fc1.weight, 1.features.14.block.2.fc1.bias |           /fc1_6/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  178  |        /activation_6/Relu       |        Relu       |                                           /fc1_6/Conv_output_0                                          |       /activation_6/Relu_output_0        |                                               |
    |  179  |           /fc2_6/Conv           |        Conv       |      /activation_6/Relu_output_0, 1.features.14.block.2.fc2.weight, 1.features.14.block.2.fc2.bias      |           /fc2_6/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  180  | /scale_activation_6/HardSigmoid |    HardSigmoid    |                                           /fc2_6/Conv_output_0                                          | /scale_activation_6/HardSigmoid_output_0 |                     alpha                     |
    |  181  |              /Mul_6             |        Mul        |                   /scale_activation_6/HardSigmoid_output_0, /block.1.2_12/Mul_output_0                  |             /Mul_6_output_0              |                                               |
    |  182  |        /block.3.0_6/Conv        |        Conv       |                             /Mul_6_output_0, onnx::Conv_647, onnx::Conv_648                             |        /block.3.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  183  |              /Add_8             |        Add        |                          /block.3.0_6/Conv_output_0, /block.3.0_5/Conv_output_0                         |             /Add_8_output_0              |                                               |
    |  184  |        /block.0.0_14/Conv       |        Conv       |                             /Add_8_output_0, onnx::Conv_650, onnx::Conv_651                             |       /block.0.0_14/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  185  |    /block.0.2_14/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_14/Conv_output_0                                       |    /block.0.2_14/HardSigmoid_output_0    |                     alpha                     |
    |  186  |        /block.0.2_14/Mul        |        Mul        |                     /block.0.0_14/Conv_output_0, /block.0.2_14/HardSigmoid_output_0                     |        /block.0.2_14/Mul_output_0        |                                               |
    |  187  |        /block.1.0_14/Conv       |        Conv       |                        /block.0.2_14/Mul_output_0, onnx::Conv_653, onnx::Conv_654                       |       /block.1.0_14/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  188  |    /block.1.2_13/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_14/Conv_output_0                                       |    /block.1.2_13/HardSigmoid_output_0    |                     alpha                     |
    |  189  |        /block.1.2_13/Mul        |        Mul        |                     /block.1.0_14/Conv_output_0, /block.1.2_13/HardSigmoid_output_0                     |        /block.1.2_13/Mul_output_0        |                                               |
    |  190  |   /avgpool_7/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_13/Mul_output_0                                       |  /avgpool_7/GlobalAveragePool_output_0   |                                               |
    |  191  |           /fc1_7/Conv           |        Conv       | /avgpool_7/GlobalAveragePool_output_0, 1.features.15.block.2.fc1.weight, 1.features.15.block.2.fc1.bias |           /fc1_7/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  192  |        /activation_7/Relu       |        Relu       |                                           /fc1_7/Conv_output_0                                          |       /activation_7/Relu_output_0        |                                               |
    |  193  |           /fc2_7/Conv           |        Conv       |      /activation_7/Relu_output_0, 1.features.15.block.2.fc2.weight, 1.features.15.block.2.fc2.bias      |           /fc2_7/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  194  | /scale_activation_7/HardSigmoid |    HardSigmoid    |                                           /fc2_7/Conv_output_0                                          | /scale_activation_7/HardSigmoid_output_0 |                     alpha                     |
    |  195  |              /Mul_7             |        Mul        |                   /scale_activation_7/HardSigmoid_output_0, /block.1.2_13/Mul_output_0                  |             /Mul_7_output_0              |                                               |
    |  196  |        /block.3.0_7/Conv        |        Conv       |                             /Mul_7_output_0, onnx::Conv_656, onnx::Conv_657                             |        /block.3.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  197  |              /Add_9             |        Add        |                               /block.3.0_7/Conv_output_0, /Add_8_output_0                               |             /Add_9_output_0              |                                               |
    |  198  |       /features.16.0/Conv       |        Conv       |                             /Add_9_output_0, onnx::Conv_659, onnx::Conv_660                             |       /features.16.0/Conv_output_0       | dilations, group, kernel_shape, pads, strides |
    |  199  |    /features.16.2/HardSigmoid   |    HardSigmoid    |                                       /features.16.0/Conv_output_0                                      |   /features.16.2/HardSigmoid_output_0    |                     alpha                     |
    |  200  |        /features.16.2/Mul       |        Mul        |                    /features.16.0/Conv_output_0, /features.16.2/HardSigmoid_output_0                    |       /features.16.2/Mul_output_0        |                                               |
    |  201  |   /avgpool_8/GlobalAveragePool  | GlobalAveragePool |                                       /features.16.2/Mul_output_0                                       |  /avgpool_8/GlobalAveragePool_output_0   |                                               |
    |  202  |             /Flatten            |      Flatten      |                                  /avgpool_8/GlobalAveragePool_output_0                                  |            /Flatten_output_0             |                      axis                     |
    |  203  |        /classifier.0/Gemm       |        Gemm       |                      /Flatten_output_0, 1.classifier.0.weight, 1.classifier.0.bias                      |       /classifier.0/Gemm_output_0        |              alpha, beta, transB              |
    |  204  |    /classifier.1/HardSigmoid    |    HardSigmoid    |                                       /classifier.0/Gemm_output_0                                       |    /classifier.1/HardSigmoid_output_0    |                     alpha                     |
    |  205  |        /classifier.1/Mul        |        Mul        |                     /classifier.0/Gemm_output_0, /classifier.1/HardSigmoid_output_0                     |        /classifier.1/Mul_output_0        |                                               |
    |  206  |        /classifier.3/Gemm       |        Gemm       |                  /classifier.1/Mul_output_0, 1.classifier.3.weight, 1.classifier.3.bias                 |                   522                    |              alpha, beta, transB              |
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+
    [32mINFO    [0m [34mQuantizing model using static quantization with calibration...[0m
    I0327 21:02:33.604533 139809641850688 quantize.py:54] Quantizing model using static quantization with calibration...
    [32mINFO    [0m [34mUsing CUDA as ONNX execution provider.[0m
    I0327 21:02:33.604749 139809641850688 utils.py:8] Using CUDA as ONNX execution provider.
    [32mINFO    [0m [34mQuantization complete. Model is now calibrated and statically quantized.[0m
    I0327 21:05:10.834060 139809641850688 quantize.py:92] Quantization complete. Model is now calibrated and statically quantized.
    [32mINFO    [0m [34mQuantizing model using dynamic quantization...[0m
    I0327 21:05:10.844158 139809641850688 quantize.py:33] Quantizing model using dynamic quantization...
    [32mINFO    [0m [34mQuantization complete. Model is now dynamically quantized.[0m
    I0327 21:05:12.147910 139809641850688 quantize.py:48] Quantization complete. Model is now dynamically quantized.
    [32mINFO    [0m [34mQuantizing model using automatic mixed precision quantization...[0m
    I0327 21:05:12.148476 139809641850688 quantize.py:98] Quantizing model using automatic mixed precision quantization...
    Adding missing dtypes for 0 outputs
    ['/0/Conv', '/features.0.0/Conv', '/features.0.2/HardSigmoid', '/features.0.2/Mul', '/block.0.0/Conv', '/block.0.2/Relu', '/block.1.0/Conv', '/Add', '/block.0.0_1/Conv', '/block.0.2_1/Relu', '/block.1.0_1/Conv', '/block.1.2/Relu', '/block.2.0/Conv', '/block.0.0_2/Conv', '/block.0.2_2/Relu', '/block.1.0_2/Conv', '/block.1.2_1/Relu', '/block.2.0_1/Conv', '/Add_1', '/block.0.0_3/Conv', '/block.0.2_3/Relu', '/block.1.0_3/Conv', '/block.1.2_2/Relu', '/avgpool/GlobalAveragePool', '/fc1/Conv', '/activation/Relu', '/fc2/Conv', '/scale_activation/HardSigmoid', '/Mul', '/block.3.0/Conv', '/block.0.0_4/Conv', '/block.0.2_4/Relu', '/block.1.0_4/Conv', '/block.1.2_3/Relu', '/avgpool_1/GlobalAveragePool', '/fc1_1/Conv', '/activation_1/Relu', '/fc2_1/Conv', '/scale_activation_1/HardSigmoid', '/Mul_1', '/block.3.0_1/Conv', '/Add_2', '/block.0.0_5/Conv', '/block.0.2_5/Relu', '/block.1.0_5/Conv', '/block.1.2_4/Relu', '/avgpool_2/GlobalAveragePool', '/fc1_2/Conv', '/activation_2/Relu', '/fc2_2/Conv', '/scale_activation_2/HardSigmoid', '/Mul_2', '/block.3.0_2/Conv', '/Add_3', '/block.0.0_6/Conv', '/block.0.2_6/HardSigmoid', '/block.0.2_6/Mul', '/block.1.0_6/Conv', '/block.1.2_5/HardSigmoid', '/block.1.2_5/Mul', '/block.2.0_2/Conv', '/block.0.0_7/Conv', '/block.0.2_7/HardSigmoid', '/block.0.2_7/Mul', '/block.1.0_7/Conv', '/block.1.2_6/HardSigmoid', '/block.1.2_6/Mul', '/block.2.0_3/Conv', '/Add_4', '/block.0.0_8/Conv', '/block.0.2_8/HardSigmoid', '/block.0.2_8/Mul', '/block.1.0_8/Conv', '/block.1.2_7/HardSigmoid', '/block.1.2_7/Mul', '/block.2.0_4/Conv', '/Add_5', '/block.0.0_9/Conv', '/block.0.2_9/HardSigmoid', '/block.0.2_9/Mul', '/block.1.0_9/Conv', '/block.1.2_8/HardSigmoid', '/block.1.2_8/Mul', '/block.2.0_5/Conv', '/Add_6', '/block.0.0_10/Conv', '/block.0.2_10/HardSigmoid', '/block.0.2_10/Mul', '/block.1.0_10/Conv', '/block.1.2_9/HardSigmoid', '/block.1.2_9/Mul', '/avgpool_3/GlobalAveragePool', '/fc1_3/Conv', '/activation_3/Relu', '/fc2_3/Conv', '/scale_activation_3/HardSigmoid', '/Mul_3', '/block.3.0_3/Conv', '/block.0.0_11/Conv', '/block.0.2_11/HardSigmoid', '/block.0.2_11/Mul', '/block.1.0_11/Conv', '/block.1.2_10/HardSigmoid', '/block.1.2_10/Mul', '/avgpool_4/GlobalAveragePool', '/fc1_4/Conv', '/activation_4/Relu', '/fc2_4/Conv', '/scale_activation_4/HardSigmoid', '/Mul_4', '/block.3.0_4/Conv', '/Add_7', '/block.0.0_12/Conv', '/block.0.2_12/HardSigmoid', '/block.0.2_12/Mul', '/block.1.0_12/Conv', '/block.1.2_11/HardSigmoid', '/block.1.2_11/Mul', '/avgpool_5/GlobalAveragePool', '/fc1_5/Conv', '/activation_5/Relu', '/fc2_5/Conv', '/scale_activation_5/HardSigmoid', '/Mul_5', '/block.3.0_5/Conv', '/block.0.0_13/Conv', '/block.0.2_13/HardSigmoid', '/block.0.2_13/Mul', '/block.1.0_13/Conv', '/block.1.2_12/HardSigmoid', '/block.1.2_12/Mul', '/avgpool_6/GlobalAveragePool', '/fc1_6/Conv', '/activation_6/Relu', '/fc2_6/Conv', '/scale_activation_6/HardSigmoid', '/Mul_6', '/block.3.0_6/Conv', '/Add_8', '/block.0.0_14/Conv', '/block.0.2_14/HardSigmoid', '/block.0.2_14/Mul', '/block.1.0_14/Conv', '/block.1.2_13/HardSigmoid', '/block.1.2_13/Mul', '/avgpool_7/GlobalAveragePool', '/fc1_7/Conv', '/activation_7/Relu', '/fc2_7/Conv', '/scale_activation_7/HardSigmoid', '/Mul_7', '/block.3.0_7/Conv', '/Add_9', '/features.16.0/Conv', '/features.16.2/HardSigmoid', '/features.16.2/Mul', '/avgpool_8/GlobalAveragePool', '/Flatten', '/classifier.0/Gemm', '/classifier.1/HardSigmoid', '/classifier.1/Mul', '/classifier.3/Gemm']
    True
    Sanity checks passed. Starting autoconvert.
    Running attempt 1 excluding conversion of 0 nodes
    []
    True
    Attempt succeeded.
    [*162*]
    Done: []
    []
    Final model validated successfully.
    [32mINFO    [0m [34mQuantization complete. Model is now quantized using automatic mixed precision.[0m
    I0327 21:05:22.048251 139809641850688 quantize.py:120] Quantization complete. Model is now quantized using automatic mixed precision.
    [32mINFO    [0m [34mPerforming runtime analysis on original graph...[0m
    I0327 21:05:22.056729 139809641850688 transform.py:170] Performing runtime analysis on original graph...
    [32mINFO    [0m [34mStarting transformation analysis on mobilenetv3_large[0m
    I0327 21:05:22.056961 139809641850688 analysis.py:276] Starting transformation analysis on mobilenetv3_large
    [32mINFO    [0m [34m
    Results mobilenetv3_large:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.10706   |
    |      Average Precision       |   0.10244   |
    |        Average Recall        |   0.11135   |
    |       Average F1 Score       |   0.10068   |
    |         Average Loss         |   2.3038    |
    |       Average Latency        |   18.7 ms   |
    |   Average GPU Power Usage    |  25.929 W   |
    | Inference Energy Consumption | 0.13469 mWh |
    +------------------------------+-------------+[0m
    I0327 21:05:28.651496 139809641850688 analysis.py:404] 
    Results mobilenetv3_large:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.10706   |
    |      Average Precision       |   0.10244   |
    |        Average Recall        |   0.11135   |
    |       Average F1 Score       |   0.10068   |
    |         Average Loss         |   2.3038    |
    |       Average Latency        |   18.7 ms   |
    |   Average GPU Power Usage    |  25.929 W   |
    | Inference Energy Consumption | 0.13469 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/mase_graph/version_0/model.json[0m
    I0327 21:05:28.652994 139809641850688 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/mase_graph/version_0/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on onnx-optimized graph...[0m
    I0327 21:05:28.653193 139809641850688 transform.py:176] Performing runtime analysis on onnx-optimized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0327 21:05:28.653386 139809641850688 analysis.py:65] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [32mINFO    [0m [34mStarting transformation analysis on mobilenetv3_large-onnx[0m
    I0327 21:05:28.896207 139809641850688 analysis.py:276] Starting transformation analysis on mobilenetv3_large-onnx
    [32mINFO    [0m [34m
    Results mobilenetv3_large-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.091344   |
    |      Average Precision       |   0.063765   |
    |        Average Recall        |   0.092105   |
    |       Average F1 Score       |   0.058955   |
    |         Average Loss         |    2.3026    |
    |       Average Latency        |  3.1337 ms   |
    |   Average GPU Power Usage    |   50.646 W   |
    | Inference Energy Consumption | 0.044086 mWh |
    +------------------------------+--------------+[0m
    I0327 21:05:33.858923 139809641850688 analysis.py:404] 
    Results mobilenetv3_large-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.091344   |
    |      Average Precision       |   0.063765   |
    |        Average Recall        |   0.092105   |
    |       Average F1 Score       |   0.058955   |
    |         Average Loss         |    2.3026    |
    |       Average Latency        |  3.1337 ms   |
    |   Average GPU Power Usage    |   50.646 W   |
    | Inference Energy Consumption | 0.044086 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_0/model.json[0m
    I0327 21:05:33.860289 139809641850688 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_0/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on static quantized graph...[0m
    I0327 21:05:33.881960 139809641850688 transform.py:191] Performing runtime analysis on static quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0327 21:05:33.882413 139809641850688 analysis.py:65] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [0;93m2024-03-27 21:05:34.235902952 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 65 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.[m
    [0;93m2024-03-27 21:05:34.241485801 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
    [0;93m2024-03-27 21:05:34.241499782 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
    [32mINFO    [0m [34mStarting transformation analysis on mobilenetv3_large-onnx[0m
    I0327 21:05:34.269716 139809641850688 analysis.py:276] Starting transformation analysis on mobilenetv3_large-onnx
    [32mINFO    [0m [34m
    Results mobilenetv3_large-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.094741   |
    |      Average Precision       |   0.060424   |
    |        Average Recall        |   0.095888   |
    |       Average F1 Score       |   0.056649   |
    |         Average Loss         |    2.3026    |
    |       Average Latency        |  4.7946 ms   |
    |   Average GPU Power Usage    |   51.271 W   |
    | Inference Energy Consumption | 0.068285 mWh |
    +------------------------------+--------------+[0m
    I0327 21:05:39.253525 139809641850688 analysis.py:404] 
    Results mobilenetv3_large-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.094741   |
    |      Average Precision       |   0.060424   |
    |        Average Recall        |   0.095888   |
    |       Average F1 Score       |   0.056649   |
    |         Average Loss         |    2.3026    |
    |       Average Latency        |  4.7946 ms   |
    |   Average GPU Power Usage    |   51.271 W   |
    | Inference Energy Consumption | 0.068285 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_1/model.json[0m
    I0327 21:05:39.255481 139809641850688 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_1/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on dynamic quantized graph...[0m
    I0327 21:05:39.324642 139809641850688 transform.py:196] Performing runtime analysis on dynamic quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0327 21:05:39.325214 139809641850688 analysis.py:65] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [0;93m2024-03-27 21:05:39.595003841 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 194 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.[m
    [0;93m2024-03-27 21:05:39.601120112 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
    [0;93m2024-03-27 21:05:39.601146467 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
    [32mINFO    [0m [34mStarting transformation analysis on mobilenetv3_large-onnx[0m
    I0327 21:05:39.625526 139809641850688 analysis.py:276] Starting transformation analysis on mobilenetv3_large-onnx
    [32mINFO    [0m [34m
    Results mobilenetv3_large-onnx:
    +------------------------------+-----------+
    |      Metric (Per Batch)      |   Value   |
    +------------------------------+-----------+
    |    Average Test Accuracy     | 0.095697  |
    |      Average Precision       |  0.06046  |
    |        Average Recall        | 0.096711  |
    |       Average F1 Score       | 0.058563  |
    |         Average Loss         |  2.3026   |
    |       Average Latency        | 666.48 ms |
    |   Average GPU Power Usage    |  23.14 W  |
    | Inference Energy Consumption | 4.284 mWh |
    +------------------------------+-----------+[0m
    I0327 21:06:50.546024 139809641850688 analysis.py:404] 
    Results mobilenetv3_large-onnx:
    +------------------------------+-----------+
    |      Metric (Per Batch)      |   Value   |
    +------------------------------+-----------+
    |    Average Test Accuracy     | 0.095697  |
    |      Average Precision       |  0.06046  |
    |        Average Recall        | 0.096711  |
    |       Average F1 Score       | 0.058563  |
    |         Average Loss         |  2.3026   |
    |       Average Latency        | 666.48 ms |
    |   Average GPU Power Usage    |  23.14 W  |
    | Inference Energy Consumption | 4.284 mWh |
    +------------------------------+-----------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_2/model.json[0m
    I0327 21:06:50.548293 139809641850688 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_2/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on auto mixed precision quantized graph...[0m
    I0327 21:06:50.582931 139809641850688 transform.py:201] Performing runtime analysis on auto mixed precision quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0327 21:06:50.583487 139809641850688 analysis.py:65] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [32mINFO    [0m [34mStarting transformation analysis on mobilenetv3_large-onnx[0m
    I0327 21:06:50.730107 139809641850688 analysis.py:276] Starting transformation analysis on mobilenetv3_large-onnx
    [32mINFO    [0m [34m
    Results mobilenetv3_large-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.10023    |
    |      Average Precision       |  0.0090688   |
    |        Average Recall        |   0.09523    |
    |       Average F1 Score       |   0.016561   |
    |         Average Loss         |    2.3026    |
    |       Average Latency        |  2.9388 ms   |
    |   Average GPU Power Usage    |   51.495 W   |
    | Inference Energy Consumption | 0.042036 mWh |
    +------------------------------+--------------+[0m
    I0327 21:06:55.735523 139809641850688 analysis.py:404] 
    Results mobilenetv3_large-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.10023    |
    |      Average Precision       |  0.0090688   |
    |        Average Recall        |   0.09523    |
    |       Average F1 Score       |   0.016561   |
    |         Average Loss         |    2.3026    |
    |       Average Latency        |  2.9388 ms   |
    |   Average GPU Power Usage    |   51.495 W   |
    | Inference Energy Consumption | 0.042036 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_3/model.json[0m
    I0327 21:06:55.737019 139809641850688 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_3/model.json
    [32mINFO    [0m [34mSaved mase graph to /root/mase/mase_output/mobilenetv3_large_cls_mnist_2024-03-27/software/transform/transformed_ckpt[0m
    I0327 21:06:55.905780 139809641850688 save_and_load.py:147] Saved mase graph to /root/mase/mase_output/mobilenetv3_large_cls_mnist_2024-03-27/software/transform/transformed_ckpt
    [32mINFO    [0m [34mTransformation is completed[0m
    I0327 21:06:55.906188 139809641850688 cli.py:383] Transformation is completed



```python
MOBILENET_TOML_PATH = "../../../machop/configs/onnx/mobilenetv3_large_gpu_quant.toml"
!ch transform --config {MOBILENET_TOML_PATH}
```

    [2024-03-27 21:20:20,076] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0327 21:20:22.271312 140179936118592 seed.py:54] Seed set to 0
    +-------------------------+------------------------+-------------------+-----------------+------------------------+
    | Name                    |        Default         |   Config. File    | Manual Override |       Effective        |
    +-------------------------+------------------------+-------------------+-----------------+------------------------+
    | task                    |     [38;5;8mclassification[0m     |        cls        |                 |          cls           |
    | load_name               |          None          |                   |                 |          None          |
    | load_type               |           mz           |                   |                 |           mz           |
    | batch_size              |          [38;5;8m128[0m           |        64         |                 |           64           |
    | to_debug                |         False          |                   |                 |         False          |
    | log_level               |          info          |                   |                 |          info          |
    | report_to               |      tensorboard       |                   |                 |      tensorboard       |
    | seed                    |           0            |                   |                 |           0            |
    | quant_config            |          None          |                   |                 |          None          |
    | training_optimizer      |          adam          |                   |                 |          adam          |
    | trainer_precision       |        16-mixed        |                   |                 |        16-mixed        |
    | learning_rate           |         [38;5;8m1e-05[0m          |       0.001       |                 |         0.001          |
    | weight_decay            |           0            |                   |                 |           0            |
    | max_epochs              |           [38;5;8m20[0m           |        10         |                 |           10           |
    | max_steps               |           -1           |                   |                 |           -1           |
    | accumulate_grad_batches |           1            |                   |                 |           1            |
    | log_every_n_steps       |           50           |                   |                 |           50           |
    | num_workers             |           28           |                   |                 |           28           |
    | num_devices             |           1            |                   |                 |           1            |
    | num_nodes               |           1            |                   |                 |           1            |
    | accelerator             |          [38;5;8mauto[0m          |        gpu        |                 |          gpu           |
    | strategy                |          auto          |                   |                 |          auto          |
    | is_to_auto_requeue      |         False          |                   |                 |         False          |
    | github_ci               |         False          |                   |                 |         False          |
    | disable_dataset_cache   |         False          |                   |                 |         False          |
    | target                  |  xcu250-figd2104-2L-e  |                   |                 |  xcu250-figd2104-2L-e  |
    | num_targets             |          100           |                   |                 |          100           |
    | is_pretrained           |         False          |                   |                 |         False          |
    | max_token_len           |          512           |                   |                 |          512           |
    | project_dir             | /root/mase/mase_output |                   |                 | /root/mase/mase_output |
    | project                 |          None          |                   |                 |          None          |
    | model                   |          [38;5;8mNone[0m          | mobilenetv3_large |                 |   mobilenetv3_large    |
    | dataset                 |          [38;5;8mNone[0m          |       mnist       |                 |         mnist          |
    | t_max                   |           20           |                   |                 |           20           |
    | eta_min                 |         1e-06          |                   |                 |         1e-06          |
    +-------------------------+------------------------+-------------------+-----------------+------------------------+
    [32mINFO    [0m [34mInitialising model 'mobilenetv3_large'...[0m
    I0327 21:20:22.279013 140179936118592 cli.py:841] Initialising model 'mobilenetv3_large'...
    [32mINFO    [0m [34mMobileNetV3 randomly initialized[0m
    I0327 21:20:22.353397 140179936118592 mobilenetv3.py:492] MobileNetV3 randomly initialized
    [32mINFO    [0m [34mInitialising dataset 'mnist'...[0m
    I0327 21:20:22.353773 140179936118592 cli.py:869] Initialising dataset 'mnist'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/mobilenetv3_large_cls_mnist_2024-03-27[0m
    I0327 21:20:22.354033 140179936118592 cli.py:905] Project will be created at /root/mase/mase_output/mobilenetv3_large_cls_mnist_2024-03-27
    [32mINFO    [0m [34mTransforming model 'mobilenetv3_large'...[0m
    I0327 21:20:22.470489 140179936118592 cli.py:365] Transforming model 'mobilenetv3_large'...
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0327 21:20:41.045079 140179936118592 onnx_runtime.py:48] Converting PyTorch model to ONNX...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/onnxrt/mobilenetv3_large_cls_mnist_2024-03-27[0m
    I0327 21:20:41.045724 140179936118592 onnx_runtime.py:50] Project will be created at /root/mase/mase_output/onnxrt/mobilenetv3_large_cls_mnist_2024-03-27
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/mobilenetv3_large_cls_mnist_2024-03-27/optimized/version_2/model.onnx[0m
    I0327 21:20:45.101868 140179936118592 onnx_runtime.py:68] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/mobilenetv3_large_cls_mnist_2024-03-27/optimized/version_2/model.onnx
    [32mINFO    [0m [34mONNX Model Summary: 
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+
    | Index |               Name              |        Type       |                                                  Inputs                                                 |                 Outputs                  |                   Attributes                  |
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+
    |   0   |            Identity_0           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_660              |                                               |
    |   1   |            Identity_1           |      Identity     |                                              onnx::Conv_639                                             |              onnx::Conv_657              |                                               |
    |   2   |            Identity_2           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_654              |                                               |
    |   3   |            Identity_3           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_651              |                                               |
    |   4   |            Identity_4           |      Identity     |                                              onnx::Conv_639                                             |              onnx::Conv_648              |                                               |
    |   5   |            Identity_5           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_645              |                                               |
    |   6   |            Identity_6           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_642              |                                               |
    |   7   |            Identity_7           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_636              |                                               |
    |   8   |            Identity_8           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_633              |                                               |
    |   9   |            Identity_9           |      Identity     |                                              onnx::Conv_621                                             |              onnx::Conv_630              |                                               |
    |   10  |           Identity_10           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_627              |                                               |
    |   11  |           Identity_11           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_624              |                                               |
    |   12  |           Identity_12           |      Identity     |                                      1.features.11.block.2.fc2.bias                                     |              onnx::Conv_618              |                                               |
    |   13  |           Identity_13           |      Identity     |                                      1.features.11.block.2.fc2.bias                                     |              onnx::Conv_615              |                                               |
    |   14  |           Identity_14           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_612              |                                               |
    |   15  |           Identity_15           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_609              |                                               |
    |   16  |           Identity_16           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_606              |                                               |
    |   17  |           Identity_17           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_603              |                                               |
    |   18  |           Identity_18           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_600              |                                               |
    |   19  |           Identity_19           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_594              |                                               |
    |   20  |           Identity_20           |      Identity     |                                              onnx::Conv_588                                             |              onnx::Conv_591              |                                               |
    |   21  |           Identity_21           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |              onnx::Conv_582              |                                               |
    |   22  |           Identity_22           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |              onnx::Conv_579              |                                               |
    |   23  |           Identity_23           |      Identity     |                                              onnx::Conv_558                                             |              onnx::Conv_576              |                                               |
    |   24  |           Identity_24           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_573              |                                               |
    |   25  |           Identity_25           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_570              |                                               |
    |   26  |           Identity_26           |      Identity     |                                              onnx::Conv_558                                             |              onnx::Conv_567              |                                               |
    |   27  |           Identity_27           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_564              |                                               |
    |   28  |           Identity_28           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_561              |                                               |
    |   29  |           Identity_29           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_555              |                                               |
    |   30  |           Identity_30           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_552              |                                               |
    |   31  |           Identity_31           |      Identity     |                                      1.features.4.block.2.fc1.bias                                      |              onnx::Conv_549              |                                               |
    |   32  |           Identity_32           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_546              |                                               |
    |   33  |           Identity_33           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_543              |                                               |
    |   34  |           Identity_34           |      Identity     |                                      1.features.4.block.2.fc1.bias                                      |              onnx::Conv_540              |                                               |
    |   35  |           Identity_35           |      Identity     |                                              onnx::Conv_534                                             |              onnx::Conv_537              |                                               |
    |   36  |           Identity_36           |      Identity     |                                              onnx::Conv_525                                             |              onnx::Conv_531              |                                               |
    |   37  |           Identity_37           |      Identity     |                                              onnx::Conv_525                                             |              onnx::Conv_528              |                                               |
    |   38  |           Identity_38           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |      1.features.15.block.2.fc2.bias      |                                               |
    |   39  |           Identity_39           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |      1.features.15.block.2.fc1.bias      |                                               |
    |   40  |           Identity_40           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |      1.features.13.block.2.fc2.bias      |                                               |
    |   41  |           Identity_41           |      Identity     |                                      1.features.12.block.2.fc1.bias                                     |      1.features.13.block.2.fc1.bias      |                                               |
    |   42  |           Identity_42           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |      1.features.11.block.2.fc1.bias      |                                               |
    |   43  |           Identity_43           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |      1.features.6.block.2.fc2.bias       |                                               |
    |   44  |           Identity_44           |      Identity     |                                      1.features.5.block.2.fc1.bias                                      |      1.features.6.block.2.fc1.bias       |                                               |
    |   45  |             /0/Conv             |        Conv       |                                         input, 0.weight, 0.bias                                         |             /0/Conv_output_0             | dilations, group, kernel_shape, pads, strides |
    |   46  |        /features.0.0/Conv       |        Conv       |                             /0/Conv_output_0, onnx::Conv_524, onnx::Conv_525                            |       /features.0.0/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   47  |    /features.0.2/HardSigmoid    |    HardSigmoid    |                                       /features.0.0/Conv_output_0                                       |    /features.0.2/HardSigmoid_output_0    |                     alpha                     |
    |   48  |        /features.0.2/Mul        |        Mul        |                     /features.0.0/Conv_output_0, /features.0.2/HardSigmoid_output_0                     |        /features.0.2/Mul_output_0        |                                               |
    |   49  |         /block.0.0/Conv         |        Conv       |                        /features.0.2/Mul_output_0, onnx::Conv_527, onnx::Conv_528                       |         /block.0.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   50  |         /block.0.2/Relu         |        Relu       |                                         /block.0.0/Conv_output_0                                        |         /block.0.2/Relu_output_0         |                                               |
    |   51  |         /block.1.0/Conv         |        Conv       |                         /block.0.2/Relu_output_0, onnx::Conv_530, onnx::Conv_531                        |         /block.1.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   52  |               /Add              |        Add        |                           /block.1.0/Conv_output_0, /features.0.2/Mul_output_0                          |              /Add_output_0               |                                               |
    |   53  |        /block.0.0_1/Conv        |        Conv       |                              /Add_output_0, onnx::Conv_533, onnx::Conv_534                              |        /block.0.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   54  |        /block.0.2_1/Relu        |        Relu       |                                        /block.0.0_1/Conv_output_0                                       |        /block.0.2_1/Relu_output_0        |                                               |
    |   55  |        /block.1.0_1/Conv        |        Conv       |                        /block.0.2_1/Relu_output_0, onnx::Conv_536, onnx::Conv_537                       |        /block.1.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   56  |         /block.1.2/Relu         |        Relu       |                                        /block.1.0_1/Conv_output_0                                       |         /block.1.2/Relu_output_0         |                                               |
    |   57  |         /block.2.0/Conv         |        Conv       |                         /block.1.2/Relu_output_0, onnx::Conv_539, onnx::Conv_540                        |         /block.2.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   58  |        /block.0.0_2/Conv        |        Conv       |                         /block.2.0/Conv_output_0, onnx::Conv_542, onnx::Conv_543                        |        /block.0.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   59  |        /block.0.2_2/Relu        |        Relu       |                                        /block.0.0_2/Conv_output_0                                       |        /block.0.2_2/Relu_output_0        |                                               |
    |   60  |        /block.1.0_2/Conv        |        Conv       |                        /block.0.2_2/Relu_output_0, onnx::Conv_545, onnx::Conv_546                       |        /block.1.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   61  |        /block.1.2_1/Relu        |        Relu       |                                        /block.1.0_2/Conv_output_0                                       |        /block.1.2_1/Relu_output_0        |                                               |
    |   62  |        /block.2.0_1/Conv        |        Conv       |                        /block.1.2_1/Relu_output_0, onnx::Conv_548, onnx::Conv_549                       |        /block.2.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   63  |              /Add_1             |        Add        |                           /block.2.0_1/Conv_output_0, /block.2.0/Conv_output_0                          |             /Add_1_output_0              |                                               |
    |   64  |        /block.0.0_3/Conv        |        Conv       |                             /Add_1_output_0, onnx::Conv_551, onnx::Conv_552                             |        /block.0.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   65  |        /block.0.2_3/Relu        |        Relu       |                                        /block.0.0_3/Conv_output_0                                       |        /block.0.2_3/Relu_output_0        |                                               |
    |   66  |        /block.1.0_3/Conv        |        Conv       |                        /block.0.2_3/Relu_output_0, onnx::Conv_554, onnx::Conv_555                       |        /block.1.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   67  |        /block.1.2_2/Relu        |        Relu       |                                        /block.1.0_3/Conv_output_0                                       |        /block.1.2_2/Relu_output_0        |                                               |
    |   68  |    /avgpool/GlobalAveragePool   | GlobalAveragePool |                                        /block.1.2_2/Relu_output_0                                       |   /avgpool/GlobalAveragePool_output_0    |                                               |
    |   69  |            /fc1/Conv            |        Conv       |   /avgpool/GlobalAveragePool_output_0, 1.features.4.block.2.fc1.weight, 1.features.4.block.2.fc1.bias   |            /fc1/Conv_output_0            | dilations, group, kernel_shape, pads, strides |
    |   70  |         /activation/Relu        |        Relu       |                                            /fc1/Conv_output_0                                           |        /activation/Relu_output_0         |                                               |
    |   71  |            /fc2/Conv            |        Conv       |        /activation/Relu_output_0, 1.features.4.block.2.fc2.weight, 1.features.4.block.2.fc2.bias        |            /fc2/Conv_output_0            | dilations, group, kernel_shape, pads, strides |
    |   72  |  /scale_activation/HardSigmoid  |    HardSigmoid    |                                            /fc2/Conv_output_0                                           |  /scale_activation/HardSigmoid_output_0  |                     alpha                     |
    |   73  |               /Mul              |        Mul        |                    /scale_activation/HardSigmoid_output_0, /block.1.2_2/Relu_output_0                   |              /Mul_output_0               |                                               |
    |   74  |         /block.3.0/Conv         |        Conv       |                              /Mul_output_0, onnx::Conv_557, onnx::Conv_558                              |         /block.3.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   75  |        /block.0.0_4/Conv        |        Conv       |                         /block.3.0/Conv_output_0, onnx::Conv_560, onnx::Conv_561                        |        /block.0.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   76  |        /block.0.2_4/Relu        |        Relu       |                                        /block.0.0_4/Conv_output_0                                       |        /block.0.2_4/Relu_output_0        |                                               |
    |   77  |        /block.1.0_4/Conv        |        Conv       |                        /block.0.2_4/Relu_output_0, onnx::Conv_563, onnx::Conv_564                       |        /block.1.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   78  |        /block.1.2_3/Relu        |        Relu       |                                        /block.1.0_4/Conv_output_0                                       |        /block.1.2_3/Relu_output_0        |                                               |
    |   79  |   /avgpool_1/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_3/Relu_output_0                                       |  /avgpool_1/GlobalAveragePool_output_0   |                                               |
    |   80  |           /fc1_1/Conv           |        Conv       |  /avgpool_1/GlobalAveragePool_output_0, 1.features.5.block.2.fc1.weight, 1.features.5.block.2.fc1.bias  |           /fc1_1/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   81  |        /activation_1/Relu       |        Relu       |                                           /fc1_1/Conv_output_0                                          |       /activation_1/Relu_output_0        |                                               |
    |   82  |           /fc2_1/Conv           |        Conv       |       /activation_1/Relu_output_0, 1.features.5.block.2.fc2.weight, 1.features.5.block.2.fc2.bias       |           /fc2_1/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   83  | /scale_activation_1/HardSigmoid |    HardSigmoid    |                                           /fc2_1/Conv_output_0                                          | /scale_activation_1/HardSigmoid_output_0 |                     alpha                     |
    |   84  |              /Mul_1             |        Mul        |                   /scale_activation_1/HardSigmoid_output_0, /block.1.2_3/Relu_output_0                  |             /Mul_1_output_0              |                                               |
    |   85  |        /block.3.0_1/Conv        |        Conv       |                             /Mul_1_output_0, onnx::Conv_566, onnx::Conv_567                             |        /block.3.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   86  |              /Add_2             |        Add        |                           /block.3.0_1/Conv_output_0, /block.3.0/Conv_output_0                          |             /Add_2_output_0              |                                               |
    |   87  |        /block.0.0_5/Conv        |        Conv       |                             /Add_2_output_0, onnx::Conv_569, onnx::Conv_570                             |        /block.0.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   88  |        /block.0.2_5/Relu        |        Relu       |                                        /block.0.0_5/Conv_output_0                                       |        /block.0.2_5/Relu_output_0        |                                               |
    |   89  |        /block.1.0_5/Conv        |        Conv       |                        /block.0.2_5/Relu_output_0, onnx::Conv_572, onnx::Conv_573                       |        /block.1.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   90  |        /block.1.2_4/Relu        |        Relu       |                                        /block.1.0_5/Conv_output_0                                       |        /block.1.2_4/Relu_output_0        |                                               |
    |   91  |   /avgpool_2/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_4/Relu_output_0                                       |  /avgpool_2/GlobalAveragePool_output_0   |                                               |
    |   92  |           /fc1_2/Conv           |        Conv       |  /avgpool_2/GlobalAveragePool_output_0, 1.features.6.block.2.fc1.weight, 1.features.6.block.2.fc1.bias  |           /fc1_2/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   93  |        /activation_2/Relu       |        Relu       |                                           /fc1_2/Conv_output_0                                          |       /activation_2/Relu_output_0        |                                               |
    |   94  |           /fc2_2/Conv           |        Conv       |       /activation_2/Relu_output_0, 1.features.6.block.2.fc2.weight, 1.features.6.block.2.fc2.bias       |           /fc2_2/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   95  | /scale_activation_2/HardSigmoid |    HardSigmoid    |                                           /fc2_2/Conv_output_0                                          | /scale_activation_2/HardSigmoid_output_0 |                     alpha                     |
    |   96  |              /Mul_2             |        Mul        |                   /scale_activation_2/HardSigmoid_output_0, /block.1.2_4/Relu_output_0                  |             /Mul_2_output_0              |                                               |
    |   97  |        /block.3.0_2/Conv        |        Conv       |                             /Mul_2_output_0, onnx::Conv_575, onnx::Conv_576                             |        /block.3.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   98  |              /Add_3             |        Add        |                               /block.3.0_2/Conv_output_0, /Add_2_output_0                               |             /Add_3_output_0              |                                               |
    |   99  |        /block.0.0_6/Conv        |        Conv       |                             /Add_3_output_0, onnx::Conv_578, onnx::Conv_579                             |        /block.0.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  100  |     /block.0.2_6/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_6/Conv_output_0                                       |    /block.0.2_6/HardSigmoid_output_0     |                     alpha                     |
    |  101  |         /block.0.2_6/Mul        |        Mul        |                      /block.0.0_6/Conv_output_0, /block.0.2_6/HardSigmoid_output_0                      |        /block.0.2_6/Mul_output_0         |                                               |
    |  102  |        /block.1.0_6/Conv        |        Conv       |                        /block.0.2_6/Mul_output_0, onnx::Conv_581, onnx::Conv_582                        |        /block.1.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  103  |     /block.1.2_5/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_6/Conv_output_0                                       |    /block.1.2_5/HardSigmoid_output_0     |                     alpha                     |
    |  104  |         /block.1.2_5/Mul        |        Mul        |                      /block.1.0_6/Conv_output_0, /block.1.2_5/HardSigmoid_output_0                      |        /block.1.2_5/Mul_output_0         |                                               |
    |  105  |        /block.2.0_2/Conv        |        Conv       |                        /block.1.2_5/Mul_output_0, onnx::Conv_584, onnx::Conv_585                        |        /block.2.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  106  |        /block.0.0_7/Conv        |        Conv       |                        /block.2.0_2/Conv_output_0, onnx::Conv_587, onnx::Conv_588                       |        /block.0.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  107  |     /block.0.2_7/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_7/Conv_output_0                                       |    /block.0.2_7/HardSigmoid_output_0     |                     alpha                     |
    |  108  |         /block.0.2_7/Mul        |        Mul        |                      /block.0.0_7/Conv_output_0, /block.0.2_7/HardSigmoid_output_0                      |        /block.0.2_7/Mul_output_0         |                                               |
    |  109  |        /block.1.0_7/Conv        |        Conv       |                        /block.0.2_7/Mul_output_0, onnx::Conv_590, onnx::Conv_591                        |        /block.1.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  110  |     /block.1.2_6/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_7/Conv_output_0                                       |    /block.1.2_6/HardSigmoid_output_0     |                     alpha                     |
    |  111  |         /block.1.2_6/Mul        |        Mul        |                      /block.1.0_7/Conv_output_0, /block.1.2_6/HardSigmoid_output_0                      |        /block.1.2_6/Mul_output_0         |                                               |
    |  112  |        /block.2.0_3/Conv        |        Conv       |                        /block.1.2_6/Mul_output_0, onnx::Conv_593, onnx::Conv_594                        |        /block.2.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  113  |              /Add_4             |        Add        |                          /block.2.0_3/Conv_output_0, /block.2.0_2/Conv_output_0                         |             /Add_4_output_0              |                                               |
    |  114  |        /block.0.0_8/Conv        |        Conv       |                             /Add_4_output_0, onnx::Conv_596, onnx::Conv_597                             |        /block.0.0_8/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  115  |     /block.0.2_8/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_8/Conv_output_0                                       |    /block.0.2_8/HardSigmoid_output_0     |                     alpha                     |
    |  116  |         /block.0.2_8/Mul        |        Mul        |                      /block.0.0_8/Conv_output_0, /block.0.2_8/HardSigmoid_output_0                      |        /block.0.2_8/Mul_output_0         |                                               |
    |  117  |        /block.1.0_8/Conv        |        Conv       |                        /block.0.2_8/Mul_output_0, onnx::Conv_599, onnx::Conv_600                        |        /block.1.0_8/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  118  |     /block.1.2_7/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_8/Conv_output_0                                       |    /block.1.2_7/HardSigmoid_output_0     |                     alpha                     |
    |  119  |         /block.1.2_7/Mul        |        Mul        |                      /block.1.0_8/Conv_output_0, /block.1.2_7/HardSigmoid_output_0                      |        /block.1.2_7/Mul_output_0         |                                               |
    |  120  |        /block.2.0_4/Conv        |        Conv       |                        /block.1.2_7/Mul_output_0, onnx::Conv_602, onnx::Conv_603                        |        /block.2.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  121  |              /Add_5             |        Add        |                               /block.2.0_4/Conv_output_0, /Add_4_output_0                               |             /Add_5_output_0              |                                               |
    |  122  |        /block.0.0_9/Conv        |        Conv       |                             /Add_5_output_0, onnx::Conv_605, onnx::Conv_606                             |        /block.0.0_9/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  123  |     /block.0.2_9/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_9/Conv_output_0                                       |    /block.0.2_9/HardSigmoid_output_0     |                     alpha                     |
    |  124  |         /block.0.2_9/Mul        |        Mul        |                      /block.0.0_9/Conv_output_0, /block.0.2_9/HardSigmoid_output_0                      |        /block.0.2_9/Mul_output_0         |                                               |
    |  125  |        /block.1.0_9/Conv        |        Conv       |                        /block.0.2_9/Mul_output_0, onnx::Conv_608, onnx::Conv_609                        |        /block.1.0_9/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  126  |     /block.1.2_8/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_9/Conv_output_0                                       |    /block.1.2_8/HardSigmoid_output_0     |                     alpha                     |
    |  127  |         /block.1.2_8/Mul        |        Mul        |                      /block.1.0_9/Conv_output_0, /block.1.2_8/HardSigmoid_output_0                      |        /block.1.2_8/Mul_output_0         |                                               |
    |  128  |        /block.2.0_5/Conv        |        Conv       |                        /block.1.2_8/Mul_output_0, onnx::Conv_611, onnx::Conv_612                        |        /block.2.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  129  |              /Add_6             |        Add        |                               /block.2.0_5/Conv_output_0, /Add_5_output_0                               |             /Add_6_output_0              |                                               |
    |  130  |        /block.0.0_10/Conv       |        Conv       |                             /Add_6_output_0, onnx::Conv_614, onnx::Conv_615                             |       /block.0.0_10/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  131  |    /block.0.2_10/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_10/Conv_output_0                                       |    /block.0.2_10/HardSigmoid_output_0    |                     alpha                     |
    |  132  |        /block.0.2_10/Mul        |        Mul        |                     /block.0.0_10/Conv_output_0, /block.0.2_10/HardSigmoid_output_0                     |        /block.0.2_10/Mul_output_0        |                                               |
    |  133  |        /block.1.0_10/Conv       |        Conv       |                        /block.0.2_10/Mul_output_0, onnx::Conv_617, onnx::Conv_618                       |       /block.1.0_10/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  134  |     /block.1.2_9/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_10/Conv_output_0                                       |    /block.1.2_9/HardSigmoid_output_0     |                     alpha                     |
    |  135  |         /block.1.2_9/Mul        |        Mul        |                      /block.1.0_10/Conv_output_0, /block.1.2_9/HardSigmoid_output_0                     |        /block.1.2_9/Mul_output_0         |                                               |
    |  136  |   /avgpool_3/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_9/Mul_output_0                                        |  /avgpool_3/GlobalAveragePool_output_0   |                                               |
    |  137  |           /fc1_3/Conv           |        Conv       | /avgpool_3/GlobalAveragePool_output_0, 1.features.11.block.2.fc1.weight, 1.features.11.block.2.fc1.bias |           /fc1_3/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  138  |        /activation_3/Relu       |        Relu       |                                           /fc1_3/Conv_output_0                                          |       /activation_3/Relu_output_0        |                                               |
    |  139  |           /fc2_3/Conv           |        Conv       |      /activation_3/Relu_output_0, 1.features.11.block.2.fc2.weight, 1.features.11.block.2.fc2.bias      |           /fc2_3/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  140  | /scale_activation_3/HardSigmoid |    HardSigmoid    |                                           /fc2_3/Conv_output_0                                          | /scale_activation_3/HardSigmoid_output_0 |                     alpha                     |
    |  141  |              /Mul_3             |        Mul        |                   /scale_activation_3/HardSigmoid_output_0, /block.1.2_9/Mul_output_0                   |             /Mul_3_output_0              |                                               |
    |  142  |        /block.3.0_3/Conv        |        Conv       |                             /Mul_3_output_0, onnx::Conv_620, onnx::Conv_621                             |        /block.3.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  143  |        /block.0.0_11/Conv       |        Conv       |                        /block.3.0_3/Conv_output_0, onnx::Conv_623, onnx::Conv_624                       |       /block.0.0_11/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  144  |    /block.0.2_11/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_11/Conv_output_0                                       |    /block.0.2_11/HardSigmoid_output_0    |                     alpha                     |
    |  145  |        /block.0.2_11/Mul        |        Mul        |                     /block.0.0_11/Conv_output_0, /block.0.2_11/HardSigmoid_output_0                     |        /block.0.2_11/Mul_output_0        |                                               |
    |  146  |        /block.1.0_11/Conv       |        Conv       |                        /block.0.2_11/Mul_output_0, onnx::Conv_626, onnx::Conv_627                       |       /block.1.0_11/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  147  |    /block.1.2_10/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_11/Conv_output_0                                       |    /block.1.2_10/HardSigmoid_output_0    |                     alpha                     |
    |  148  |        /block.1.2_10/Mul        |        Mul        |                     /block.1.0_11/Conv_output_0, /block.1.2_10/HardSigmoid_output_0                     |        /block.1.2_10/Mul_output_0        |                                               |
    |  149  |   /avgpool_4/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_10/Mul_output_0                                       |  /avgpool_4/GlobalAveragePool_output_0   |                                               |
    |  150  |           /fc1_4/Conv           |        Conv       | /avgpool_4/GlobalAveragePool_output_0, 1.features.12.block.2.fc1.weight, 1.features.12.block.2.fc1.bias |           /fc1_4/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  151  |        /activation_4/Relu       |        Relu       |                                           /fc1_4/Conv_output_0                                          |       /activation_4/Relu_output_0        |                                               |
    |  152  |           /fc2_4/Conv           |        Conv       |      /activation_4/Relu_output_0, 1.features.12.block.2.fc2.weight, 1.features.12.block.2.fc2.bias      |           /fc2_4/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  153  | /scale_activation_4/HardSigmoid |    HardSigmoid    |                                           /fc2_4/Conv_output_0                                          | /scale_activation_4/HardSigmoid_output_0 |                     alpha                     |
    |  154  |              /Mul_4             |        Mul        |                   /scale_activation_4/HardSigmoid_output_0, /block.1.2_10/Mul_output_0                  |             /Mul_4_output_0              |                                               |
    |  155  |        /block.3.0_4/Conv        |        Conv       |                             /Mul_4_output_0, onnx::Conv_629, onnx::Conv_630                             |        /block.3.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  156  |              /Add_7             |        Add        |                          /block.3.0_4/Conv_output_0, /block.3.0_3/Conv_output_0                         |             /Add_7_output_0              |                                               |
    |  157  |        /block.0.0_12/Conv       |        Conv       |                             /Add_7_output_0, onnx::Conv_632, onnx::Conv_633                             |       /block.0.0_12/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  158  |    /block.0.2_12/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_12/Conv_output_0                                       |    /block.0.2_12/HardSigmoid_output_0    |                     alpha                     |
    |  159  |        /block.0.2_12/Mul        |        Mul        |                     /block.0.0_12/Conv_output_0, /block.0.2_12/HardSigmoid_output_0                     |        /block.0.2_12/Mul_output_0        |                                               |
    |  160  |        /block.1.0_12/Conv       |        Conv       |                        /block.0.2_12/Mul_output_0, onnx::Conv_635, onnx::Conv_636                       |       /block.1.0_12/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  161  |    /block.1.2_11/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_12/Conv_output_0                                       |    /block.1.2_11/HardSigmoid_output_0    |                     alpha                     |
    |  162  |        /block.1.2_11/Mul        |        Mul        |                     /block.1.0_12/Conv_output_0, /block.1.2_11/HardSigmoid_output_0                     |        /block.1.2_11/Mul_output_0        |                                               |
    |  163  |   /avgpool_5/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_11/Mul_output_0                                       |  /avgpool_5/GlobalAveragePool_output_0   |                                               |
    |  164  |           /fc1_5/Conv           |        Conv       | /avgpool_5/GlobalAveragePool_output_0, 1.features.13.block.2.fc1.weight, 1.features.13.block.2.fc1.bias |           /fc1_5/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  165  |        /activation_5/Relu       |        Relu       |                                           /fc1_5/Conv_output_0                                          |       /activation_5/Relu_output_0        |                                               |
    |  166  |           /fc2_5/Conv           |        Conv       |      /activation_5/Relu_output_0, 1.features.13.block.2.fc2.weight, 1.features.13.block.2.fc2.bias      |           /fc2_5/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  167  | /scale_activation_5/HardSigmoid |    HardSigmoid    |                                           /fc2_5/Conv_output_0                                          | /scale_activation_5/HardSigmoid_output_0 |                     alpha                     |
    |  168  |              /Mul_5             |        Mul        |                   /scale_activation_5/HardSigmoid_output_0, /block.1.2_11/Mul_output_0                  |             /Mul_5_output_0              |                                               |
    |  169  |        /block.3.0_5/Conv        |        Conv       |                             /Mul_5_output_0, onnx::Conv_638, onnx::Conv_639                             |        /block.3.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  170  |        /block.0.0_13/Conv       |        Conv       |                        /block.3.0_5/Conv_output_0, onnx::Conv_641, onnx::Conv_642                       |       /block.0.0_13/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  171  |    /block.0.2_13/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_13/Conv_output_0                                       |    /block.0.2_13/HardSigmoid_output_0    |                     alpha                     |
    |  172  |        /block.0.2_13/Mul        |        Mul        |                     /block.0.0_13/Conv_output_0, /block.0.2_13/HardSigmoid_output_0                     |        /block.0.2_13/Mul_output_0        |                                               |
    |  173  |        /block.1.0_13/Conv       |        Conv       |                        /block.0.2_13/Mul_output_0, onnx::Conv_644, onnx::Conv_645                       |       /block.1.0_13/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  174  |    /block.1.2_12/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_13/Conv_output_0                                       |    /block.1.2_12/HardSigmoid_output_0    |                     alpha                     |
    |  175  |        /block.1.2_12/Mul        |        Mul        |                     /block.1.0_13/Conv_output_0, /block.1.2_12/HardSigmoid_output_0                     |        /block.1.2_12/Mul_output_0        |                                               |
    |  176  |   /avgpool_6/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_12/Mul_output_0                                       |  /avgpool_6/GlobalAveragePool_output_0   |                                               |
    |  177  |           /fc1_6/Conv           |        Conv       | /avgpool_6/GlobalAveragePool_output_0, 1.features.14.block.2.fc1.weight, 1.features.14.block.2.fc1.bias |           /fc1_6/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  178  |        /activation_6/Relu       |        Relu       |                                           /fc1_6/Conv_output_0                                          |       /activation_6/Relu_output_0        |                                               |
    |  179  |           /fc2_6/Conv           |        Conv       |      /activation_6/Relu_output_0, 1.features.14.block.2.fc2.weight, 1.features.14.block.2.fc2.bias      |           /fc2_6/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  180  | /scale_activation_6/HardSigmoid |    HardSigmoid    |                                           /fc2_6/Conv_output_0                                          | /scale_activation_6/HardSigmoid_output_0 |                     alpha                     |
    |  181  |              /Mul_6             |        Mul        |                   /scale_activation_6/HardSigmoid_output_0, /block.1.2_12/Mul_output_0                  |             /Mul_6_output_0              |                                               |
    |  182  |        /block.3.0_6/Conv        |        Conv       |                             /Mul_6_output_0, onnx::Conv_647, onnx::Conv_648                             |        /block.3.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  183  |              /Add_8             |        Add        |                          /block.3.0_6/Conv_output_0, /block.3.0_5/Conv_output_0                         |             /Add_8_output_0              |                                               |
    |  184  |        /block.0.0_14/Conv       |        Conv       |                             /Add_8_output_0, onnx::Conv_650, onnx::Conv_651                             |       /block.0.0_14/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  185  |    /block.0.2_14/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_14/Conv_output_0                                       |    /block.0.2_14/HardSigmoid_output_0    |                     alpha                     |
    |  186  |        /block.0.2_14/Mul        |        Mul        |                     /block.0.0_14/Conv_output_0, /block.0.2_14/HardSigmoid_output_0                     |        /block.0.2_14/Mul_output_0        |                                               |
    |  187  |        /block.1.0_14/Conv       |        Conv       |                        /block.0.2_14/Mul_output_0, onnx::Conv_653, onnx::Conv_654                       |       /block.1.0_14/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  188  |    /block.1.2_13/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_14/Conv_output_0                                       |    /block.1.2_13/HardSigmoid_output_0    |                     alpha                     |
    |  189  |        /block.1.2_13/Mul        |        Mul        |                     /block.1.0_14/Conv_output_0, /block.1.2_13/HardSigmoid_output_0                     |        /block.1.2_13/Mul_output_0        |                                               |
    |  190  |   /avgpool_7/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_13/Mul_output_0                                       |  /avgpool_7/GlobalAveragePool_output_0   |                                               |
    |  191  |           /fc1_7/Conv           |        Conv       | /avgpool_7/GlobalAveragePool_output_0, 1.features.15.block.2.fc1.weight, 1.features.15.block.2.fc1.bias |           /fc1_7/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  192  |        /activation_7/Relu       |        Relu       |                                           /fc1_7/Conv_output_0                                          |       /activation_7/Relu_output_0        |                                               |
    |  193  |           /fc2_7/Conv           |        Conv       |      /activation_7/Relu_output_0, 1.features.15.block.2.fc2.weight, 1.features.15.block.2.fc2.bias      |           /fc2_7/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  194  | /scale_activation_7/HardSigmoid |    HardSigmoid    |                                           /fc2_7/Conv_output_0                                          | /scale_activation_7/HardSigmoid_output_0 |                     alpha                     |
    |  195  |              /Mul_7             |        Mul        |                   /scale_activation_7/HardSigmoid_output_0, /block.1.2_13/Mul_output_0                  |             /Mul_7_output_0              |                                               |
    |  196  |        /block.3.0_7/Conv        |        Conv       |                             /Mul_7_output_0, onnx::Conv_656, onnx::Conv_657                             |        /block.3.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  197  |              /Add_9             |        Add        |                               /block.3.0_7/Conv_output_0, /Add_8_output_0                               |             /Add_9_output_0              |                                               |
    |  198  |       /features.16.0/Conv       |        Conv       |                             /Add_9_output_0, onnx::Conv_659, onnx::Conv_660                             |       /features.16.0/Conv_output_0       | dilations, group, kernel_shape, pads, strides |
    |  199  |    /features.16.2/HardSigmoid   |    HardSigmoid    |                                       /features.16.0/Conv_output_0                                      |   /features.16.2/HardSigmoid_output_0    |                     alpha                     |
    |  200  |        /features.16.2/Mul       |        Mul        |                    /features.16.0/Conv_output_0, /features.16.2/HardSigmoid_output_0                    |       /features.16.2/Mul_output_0        |                                               |
    |  201  |   /avgpool_8/GlobalAveragePool  | GlobalAveragePool |                                       /features.16.2/Mul_output_0                                       |  /avgpool_8/GlobalAveragePool_output_0   |                                               |
    |  202  |             /Flatten            |      Flatten      |                                  /avgpool_8/GlobalAveragePool_output_0                                  |            /Flatten_output_0             |                      axis                     |
    |  203  |        /classifier.0/Gemm       |        Gemm       |                      /Flatten_output_0, 1.classifier.0.weight, 1.classifier.0.bias                      |       /classifier.0/Gemm_output_0        |              alpha, beta, transB              |
    |  204  |    /classifier.1/HardSigmoid    |    HardSigmoid    |                                       /classifier.0/Gemm_output_0                                       |    /classifier.1/HardSigmoid_output_0    |                     alpha                     |
    |  205  |        /classifier.1/Mul        |        Mul        |                     /classifier.0/Gemm_output_0, /classifier.1/HardSigmoid_output_0                     |        /classifier.1/Mul_output_0        |                                               |
    |  206  |        /classifier.3/Gemm       |        Gemm       |                  /classifier.1/Mul_output_0, 1.classifier.3.weight, 1.classifier.3.bias                 |                   522                    |              alpha, beta, transB              |
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+[0m
    I0327 21:20:45.168224 140179936118592 onnx_runtime.py:90] ONNX Model Summary: 
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+
    | Index |               Name              |        Type       |                                                  Inputs                                                 |                 Outputs                  |                   Attributes                  |
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+
    |   0   |            Identity_0           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_660              |                                               |
    |   1   |            Identity_1           |      Identity     |                                              onnx::Conv_639                                             |              onnx::Conv_657              |                                               |
    |   2   |            Identity_2           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_654              |                                               |
    |   3   |            Identity_3           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_651              |                                               |
    |   4   |            Identity_4           |      Identity     |                                              onnx::Conv_639                                             |              onnx::Conv_648              |                                               |
    |   5   |            Identity_5           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_645              |                                               |
    |   6   |            Identity_6           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |              onnx::Conv_642              |                                               |
    |   7   |            Identity_7           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_636              |                                               |
    |   8   |            Identity_8           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_633              |                                               |
    |   9   |            Identity_9           |      Identity     |                                              onnx::Conv_621                                             |              onnx::Conv_630              |                                               |
    |   10  |           Identity_10           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_627              |                                               |
    |   11  |           Identity_11           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |              onnx::Conv_624              |                                               |
    |   12  |           Identity_12           |      Identity     |                                      1.features.11.block.2.fc2.bias                                     |              onnx::Conv_618              |                                               |
    |   13  |           Identity_13           |      Identity     |                                      1.features.11.block.2.fc2.bias                                     |              onnx::Conv_615              |                                               |
    |   14  |           Identity_14           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_612              |                                               |
    |   15  |           Identity_15           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_609              |                                               |
    |   16  |           Identity_16           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_606              |                                               |
    |   17  |           Identity_17           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_603              |                                               |
    |   18  |           Identity_18           |      Identity     |                                              onnx::Conv_597                                             |              onnx::Conv_600              |                                               |
    |   19  |           Identity_19           |      Identity     |                                              onnx::Conv_585                                             |              onnx::Conv_594              |                                               |
    |   20  |           Identity_20           |      Identity     |                                              onnx::Conv_588                                             |              onnx::Conv_591              |                                               |
    |   21  |           Identity_21           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |              onnx::Conv_582              |                                               |
    |   22  |           Identity_22           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |              onnx::Conv_579              |                                               |
    |   23  |           Identity_23           |      Identity     |                                              onnx::Conv_558                                             |              onnx::Conv_576              |                                               |
    |   24  |           Identity_24           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_573              |                                               |
    |   25  |           Identity_25           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_570              |                                               |
    |   26  |           Identity_26           |      Identity     |                                              onnx::Conv_558                                             |              onnx::Conv_567              |                                               |
    |   27  |           Identity_27           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_564              |                                               |
    |   28  |           Identity_28           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |              onnx::Conv_561              |                                               |
    |   29  |           Identity_29           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_555              |                                               |
    |   30  |           Identity_30           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_552              |                                               |
    |   31  |           Identity_31           |      Identity     |                                      1.features.4.block.2.fc1.bias                                      |              onnx::Conv_549              |                                               |
    |   32  |           Identity_32           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_546              |                                               |
    |   33  |           Identity_33           |      Identity     |                                      1.features.4.block.2.fc2.bias                                      |              onnx::Conv_543              |                                               |
    |   34  |           Identity_34           |      Identity     |                                      1.features.4.block.2.fc1.bias                                      |              onnx::Conv_540              |                                               |
    |   35  |           Identity_35           |      Identity     |                                              onnx::Conv_534                                             |              onnx::Conv_537              |                                               |
    |   36  |           Identity_36           |      Identity     |                                              onnx::Conv_525                                             |              onnx::Conv_531              |                                               |
    |   37  |           Identity_37           |      Identity     |                                              onnx::Conv_525                                             |              onnx::Conv_528              |                                               |
    |   38  |           Identity_38           |      Identity     |                                      1.features.14.block.2.fc2.bias                                     |      1.features.15.block.2.fc2.bias      |                                               |
    |   39  |           Identity_39           |      Identity     |                                      1.features.14.block.2.fc1.bias                                     |      1.features.15.block.2.fc1.bias      |                                               |
    |   40  |           Identity_40           |      Identity     |                                      1.features.12.block.2.fc2.bias                                     |      1.features.13.block.2.fc2.bias      |                                               |
    |   41  |           Identity_41           |      Identity     |                                      1.features.12.block.2.fc1.bias                                     |      1.features.13.block.2.fc1.bias      |                                               |
    |   42  |           Identity_42           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |      1.features.11.block.2.fc1.bias      |                                               |
    |   43  |           Identity_43           |      Identity     |                                      1.features.5.block.2.fc2.bias                                      |      1.features.6.block.2.fc2.bias       |                                               |
    |   44  |           Identity_44           |      Identity     |                                      1.features.5.block.2.fc1.bias                                      |      1.features.6.block.2.fc1.bias       |                                               |
    |   45  |             /0/Conv             |        Conv       |                                         input, 0.weight, 0.bias                                         |             /0/Conv_output_0             | dilations, group, kernel_shape, pads, strides |
    |   46  |        /features.0.0/Conv       |        Conv       |                             /0/Conv_output_0, onnx::Conv_524, onnx::Conv_525                            |       /features.0.0/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   47  |    /features.0.2/HardSigmoid    |    HardSigmoid    |                                       /features.0.0/Conv_output_0                                       |    /features.0.2/HardSigmoid_output_0    |                     alpha                     |
    |   48  |        /features.0.2/Mul        |        Mul        |                     /features.0.0/Conv_output_0, /features.0.2/HardSigmoid_output_0                     |        /features.0.2/Mul_output_0        |                                               |
    |   49  |         /block.0.0/Conv         |        Conv       |                        /features.0.2/Mul_output_0, onnx::Conv_527, onnx::Conv_528                       |         /block.0.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   50  |         /block.0.2/Relu         |        Relu       |                                         /block.0.0/Conv_output_0                                        |         /block.0.2/Relu_output_0         |                                               |
    |   51  |         /block.1.0/Conv         |        Conv       |                         /block.0.2/Relu_output_0, onnx::Conv_530, onnx::Conv_531                        |         /block.1.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   52  |               /Add              |        Add        |                           /block.1.0/Conv_output_0, /features.0.2/Mul_output_0                          |              /Add_output_0               |                                               |
    |   53  |        /block.0.0_1/Conv        |        Conv       |                              /Add_output_0, onnx::Conv_533, onnx::Conv_534                              |        /block.0.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   54  |        /block.0.2_1/Relu        |        Relu       |                                        /block.0.0_1/Conv_output_0                                       |        /block.0.2_1/Relu_output_0        |                                               |
    |   55  |        /block.1.0_1/Conv        |        Conv       |                        /block.0.2_1/Relu_output_0, onnx::Conv_536, onnx::Conv_537                       |        /block.1.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   56  |         /block.1.2/Relu         |        Relu       |                                        /block.1.0_1/Conv_output_0                                       |         /block.1.2/Relu_output_0         |                                               |
    |   57  |         /block.2.0/Conv         |        Conv       |                         /block.1.2/Relu_output_0, onnx::Conv_539, onnx::Conv_540                        |         /block.2.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   58  |        /block.0.0_2/Conv        |        Conv       |                         /block.2.0/Conv_output_0, onnx::Conv_542, onnx::Conv_543                        |        /block.0.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   59  |        /block.0.2_2/Relu        |        Relu       |                                        /block.0.0_2/Conv_output_0                                       |        /block.0.2_2/Relu_output_0        |                                               |
    |   60  |        /block.1.0_2/Conv        |        Conv       |                        /block.0.2_2/Relu_output_0, onnx::Conv_545, onnx::Conv_546                       |        /block.1.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   61  |        /block.1.2_1/Relu        |        Relu       |                                        /block.1.0_2/Conv_output_0                                       |        /block.1.2_1/Relu_output_0        |                                               |
    |   62  |        /block.2.0_1/Conv        |        Conv       |                        /block.1.2_1/Relu_output_0, onnx::Conv_548, onnx::Conv_549                       |        /block.2.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   63  |              /Add_1             |        Add        |                           /block.2.0_1/Conv_output_0, /block.2.0/Conv_output_0                          |             /Add_1_output_0              |                                               |
    |   64  |        /block.0.0_3/Conv        |        Conv       |                             /Add_1_output_0, onnx::Conv_551, onnx::Conv_552                             |        /block.0.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   65  |        /block.0.2_3/Relu        |        Relu       |                                        /block.0.0_3/Conv_output_0                                       |        /block.0.2_3/Relu_output_0        |                                               |
    |   66  |        /block.1.0_3/Conv        |        Conv       |                        /block.0.2_3/Relu_output_0, onnx::Conv_554, onnx::Conv_555                       |        /block.1.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   67  |        /block.1.2_2/Relu        |        Relu       |                                        /block.1.0_3/Conv_output_0                                       |        /block.1.2_2/Relu_output_0        |                                               |
    |   68  |    /avgpool/GlobalAveragePool   | GlobalAveragePool |                                        /block.1.2_2/Relu_output_0                                       |   /avgpool/GlobalAveragePool_output_0    |                                               |
    |   69  |            /fc1/Conv            |        Conv       |   /avgpool/GlobalAveragePool_output_0, 1.features.4.block.2.fc1.weight, 1.features.4.block.2.fc1.bias   |            /fc1/Conv_output_0            | dilations, group, kernel_shape, pads, strides |
    |   70  |         /activation/Relu        |        Relu       |                                            /fc1/Conv_output_0                                           |        /activation/Relu_output_0         |                                               |
    |   71  |            /fc2/Conv            |        Conv       |        /activation/Relu_output_0, 1.features.4.block.2.fc2.weight, 1.features.4.block.2.fc2.bias        |            /fc2/Conv_output_0            | dilations, group, kernel_shape, pads, strides |
    |   72  |  /scale_activation/HardSigmoid  |    HardSigmoid    |                                            /fc2/Conv_output_0                                           |  /scale_activation/HardSigmoid_output_0  |                     alpha                     |
    |   73  |               /Mul              |        Mul        |                    /scale_activation/HardSigmoid_output_0, /block.1.2_2/Relu_output_0                   |              /Mul_output_0               |                                               |
    |   74  |         /block.3.0/Conv         |        Conv       |                              /Mul_output_0, onnx::Conv_557, onnx::Conv_558                              |         /block.3.0/Conv_output_0         | dilations, group, kernel_shape, pads, strides |
    |   75  |        /block.0.0_4/Conv        |        Conv       |                         /block.3.0/Conv_output_0, onnx::Conv_560, onnx::Conv_561                        |        /block.0.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   76  |        /block.0.2_4/Relu        |        Relu       |                                        /block.0.0_4/Conv_output_0                                       |        /block.0.2_4/Relu_output_0        |                                               |
    |   77  |        /block.1.0_4/Conv        |        Conv       |                        /block.0.2_4/Relu_output_0, onnx::Conv_563, onnx::Conv_564                       |        /block.1.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   78  |        /block.1.2_3/Relu        |        Relu       |                                        /block.1.0_4/Conv_output_0                                       |        /block.1.2_3/Relu_output_0        |                                               |
    |   79  |   /avgpool_1/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_3/Relu_output_0                                       |  /avgpool_1/GlobalAveragePool_output_0   |                                               |
    |   80  |           /fc1_1/Conv           |        Conv       |  /avgpool_1/GlobalAveragePool_output_0, 1.features.5.block.2.fc1.weight, 1.features.5.block.2.fc1.bias  |           /fc1_1/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   81  |        /activation_1/Relu       |        Relu       |                                           /fc1_1/Conv_output_0                                          |       /activation_1/Relu_output_0        |                                               |
    |   82  |           /fc2_1/Conv           |        Conv       |       /activation_1/Relu_output_0, 1.features.5.block.2.fc2.weight, 1.features.5.block.2.fc2.bias       |           /fc2_1/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   83  | /scale_activation_1/HardSigmoid |    HardSigmoid    |                                           /fc2_1/Conv_output_0                                          | /scale_activation_1/HardSigmoid_output_0 |                     alpha                     |
    |   84  |              /Mul_1             |        Mul        |                   /scale_activation_1/HardSigmoid_output_0, /block.1.2_3/Relu_output_0                  |             /Mul_1_output_0              |                                               |
    |   85  |        /block.3.0_1/Conv        |        Conv       |                             /Mul_1_output_0, onnx::Conv_566, onnx::Conv_567                             |        /block.3.0_1/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   86  |              /Add_2             |        Add        |                           /block.3.0_1/Conv_output_0, /block.3.0/Conv_output_0                          |             /Add_2_output_0              |                                               |
    |   87  |        /block.0.0_5/Conv        |        Conv       |                             /Add_2_output_0, onnx::Conv_569, onnx::Conv_570                             |        /block.0.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   88  |        /block.0.2_5/Relu        |        Relu       |                                        /block.0.0_5/Conv_output_0                                       |        /block.0.2_5/Relu_output_0        |                                               |
    |   89  |        /block.1.0_5/Conv        |        Conv       |                        /block.0.2_5/Relu_output_0, onnx::Conv_572, onnx::Conv_573                       |        /block.1.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   90  |        /block.1.2_4/Relu        |        Relu       |                                        /block.1.0_5/Conv_output_0                                       |        /block.1.2_4/Relu_output_0        |                                               |
    |   91  |   /avgpool_2/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_4/Relu_output_0                                       |  /avgpool_2/GlobalAveragePool_output_0   |                                               |
    |   92  |           /fc1_2/Conv           |        Conv       |  /avgpool_2/GlobalAveragePool_output_0, 1.features.6.block.2.fc1.weight, 1.features.6.block.2.fc1.bias  |           /fc1_2/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   93  |        /activation_2/Relu       |        Relu       |                                           /fc1_2/Conv_output_0                                          |       /activation_2/Relu_output_0        |                                               |
    |   94  |           /fc2_2/Conv           |        Conv       |       /activation_2/Relu_output_0, 1.features.6.block.2.fc2.weight, 1.features.6.block.2.fc2.bias       |           /fc2_2/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |   95  | /scale_activation_2/HardSigmoid |    HardSigmoid    |                                           /fc2_2/Conv_output_0                                          | /scale_activation_2/HardSigmoid_output_0 |                     alpha                     |
    |   96  |              /Mul_2             |        Mul        |                   /scale_activation_2/HardSigmoid_output_0, /block.1.2_4/Relu_output_0                  |             /Mul_2_output_0              |                                               |
    |   97  |        /block.3.0_2/Conv        |        Conv       |                             /Mul_2_output_0, onnx::Conv_575, onnx::Conv_576                             |        /block.3.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |   98  |              /Add_3             |        Add        |                               /block.3.0_2/Conv_output_0, /Add_2_output_0                               |             /Add_3_output_0              |                                               |
    |   99  |        /block.0.0_6/Conv        |        Conv       |                             /Add_3_output_0, onnx::Conv_578, onnx::Conv_579                             |        /block.0.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  100  |     /block.0.2_6/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_6/Conv_output_0                                       |    /block.0.2_6/HardSigmoid_output_0     |                     alpha                     |
    |  101  |         /block.0.2_6/Mul        |        Mul        |                      /block.0.0_6/Conv_output_0, /block.0.2_6/HardSigmoid_output_0                      |        /block.0.2_6/Mul_output_0         |                                               |
    |  102  |        /block.1.0_6/Conv        |        Conv       |                        /block.0.2_6/Mul_output_0, onnx::Conv_581, onnx::Conv_582                        |        /block.1.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  103  |     /block.1.2_5/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_6/Conv_output_0                                       |    /block.1.2_5/HardSigmoid_output_0     |                     alpha                     |
    |  104  |         /block.1.2_5/Mul        |        Mul        |                      /block.1.0_6/Conv_output_0, /block.1.2_5/HardSigmoid_output_0                      |        /block.1.2_5/Mul_output_0         |                                               |
    |  105  |        /block.2.0_2/Conv        |        Conv       |                        /block.1.2_5/Mul_output_0, onnx::Conv_584, onnx::Conv_585                        |        /block.2.0_2/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  106  |        /block.0.0_7/Conv        |        Conv       |                        /block.2.0_2/Conv_output_0, onnx::Conv_587, onnx::Conv_588                       |        /block.0.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  107  |     /block.0.2_7/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_7/Conv_output_0                                       |    /block.0.2_7/HardSigmoid_output_0     |                     alpha                     |
    |  108  |         /block.0.2_7/Mul        |        Mul        |                      /block.0.0_7/Conv_output_0, /block.0.2_7/HardSigmoid_output_0                      |        /block.0.2_7/Mul_output_0         |                                               |
    |  109  |        /block.1.0_7/Conv        |        Conv       |                        /block.0.2_7/Mul_output_0, onnx::Conv_590, onnx::Conv_591                        |        /block.1.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  110  |     /block.1.2_6/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_7/Conv_output_0                                       |    /block.1.2_6/HardSigmoid_output_0     |                     alpha                     |
    |  111  |         /block.1.2_6/Mul        |        Mul        |                      /block.1.0_7/Conv_output_0, /block.1.2_6/HardSigmoid_output_0                      |        /block.1.2_6/Mul_output_0         |                                               |
    |  112  |        /block.2.0_3/Conv        |        Conv       |                        /block.1.2_6/Mul_output_0, onnx::Conv_593, onnx::Conv_594                        |        /block.2.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  113  |              /Add_4             |        Add        |                          /block.2.0_3/Conv_output_0, /block.2.0_2/Conv_output_0                         |             /Add_4_output_0              |                                               |
    |  114  |        /block.0.0_8/Conv        |        Conv       |                             /Add_4_output_0, onnx::Conv_596, onnx::Conv_597                             |        /block.0.0_8/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  115  |     /block.0.2_8/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_8/Conv_output_0                                       |    /block.0.2_8/HardSigmoid_output_0     |                     alpha                     |
    |  116  |         /block.0.2_8/Mul        |        Mul        |                      /block.0.0_8/Conv_output_0, /block.0.2_8/HardSigmoid_output_0                      |        /block.0.2_8/Mul_output_0         |                                               |
    |  117  |        /block.1.0_8/Conv        |        Conv       |                        /block.0.2_8/Mul_output_0, onnx::Conv_599, onnx::Conv_600                        |        /block.1.0_8/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  118  |     /block.1.2_7/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_8/Conv_output_0                                       |    /block.1.2_7/HardSigmoid_output_0     |                     alpha                     |
    |  119  |         /block.1.2_7/Mul        |        Mul        |                      /block.1.0_8/Conv_output_0, /block.1.2_7/HardSigmoid_output_0                      |        /block.1.2_7/Mul_output_0         |                                               |
    |  120  |        /block.2.0_4/Conv        |        Conv       |                        /block.1.2_7/Mul_output_0, onnx::Conv_602, onnx::Conv_603                        |        /block.2.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  121  |              /Add_5             |        Add        |                               /block.2.0_4/Conv_output_0, /Add_4_output_0                               |             /Add_5_output_0              |                                               |
    |  122  |        /block.0.0_9/Conv        |        Conv       |                             /Add_5_output_0, onnx::Conv_605, onnx::Conv_606                             |        /block.0.0_9/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  123  |     /block.0.2_9/HardSigmoid    |    HardSigmoid    |                                        /block.0.0_9/Conv_output_0                                       |    /block.0.2_9/HardSigmoid_output_0     |                     alpha                     |
    |  124  |         /block.0.2_9/Mul        |        Mul        |                      /block.0.0_9/Conv_output_0, /block.0.2_9/HardSigmoid_output_0                      |        /block.0.2_9/Mul_output_0         |                                               |
    |  125  |        /block.1.0_9/Conv        |        Conv       |                        /block.0.2_9/Mul_output_0, onnx::Conv_608, onnx::Conv_609                        |        /block.1.0_9/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  126  |     /block.1.2_8/HardSigmoid    |    HardSigmoid    |                                        /block.1.0_9/Conv_output_0                                       |    /block.1.2_8/HardSigmoid_output_0     |                     alpha                     |
    |  127  |         /block.1.2_8/Mul        |        Mul        |                      /block.1.0_9/Conv_output_0, /block.1.2_8/HardSigmoid_output_0                      |        /block.1.2_8/Mul_output_0         |                                               |
    |  128  |        /block.2.0_5/Conv        |        Conv       |                        /block.1.2_8/Mul_output_0, onnx::Conv_611, onnx::Conv_612                        |        /block.2.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  129  |              /Add_6             |        Add        |                               /block.2.0_5/Conv_output_0, /Add_5_output_0                               |             /Add_6_output_0              |                                               |
    |  130  |        /block.0.0_10/Conv       |        Conv       |                             /Add_6_output_0, onnx::Conv_614, onnx::Conv_615                             |       /block.0.0_10/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  131  |    /block.0.2_10/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_10/Conv_output_0                                       |    /block.0.2_10/HardSigmoid_output_0    |                     alpha                     |
    |  132  |        /block.0.2_10/Mul        |        Mul        |                     /block.0.0_10/Conv_output_0, /block.0.2_10/HardSigmoid_output_0                     |        /block.0.2_10/Mul_output_0        |                                               |
    |  133  |        /block.1.0_10/Conv       |        Conv       |                        /block.0.2_10/Mul_output_0, onnx::Conv_617, onnx::Conv_618                       |       /block.1.0_10/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  134  |     /block.1.2_9/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_10/Conv_output_0                                       |    /block.1.2_9/HardSigmoid_output_0     |                     alpha                     |
    |  135  |         /block.1.2_9/Mul        |        Mul        |                      /block.1.0_10/Conv_output_0, /block.1.2_9/HardSigmoid_output_0                     |        /block.1.2_9/Mul_output_0         |                                               |
    |  136  |   /avgpool_3/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_9/Mul_output_0                                        |  /avgpool_3/GlobalAveragePool_output_0   |                                               |
    |  137  |           /fc1_3/Conv           |        Conv       | /avgpool_3/GlobalAveragePool_output_0, 1.features.11.block.2.fc1.weight, 1.features.11.block.2.fc1.bias |           /fc1_3/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  138  |        /activation_3/Relu       |        Relu       |                                           /fc1_3/Conv_output_0                                          |       /activation_3/Relu_output_0        |                                               |
    |  139  |           /fc2_3/Conv           |        Conv       |      /activation_3/Relu_output_0, 1.features.11.block.2.fc2.weight, 1.features.11.block.2.fc2.bias      |           /fc2_3/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  140  | /scale_activation_3/HardSigmoid |    HardSigmoid    |                                           /fc2_3/Conv_output_0                                          | /scale_activation_3/HardSigmoid_output_0 |                     alpha                     |
    |  141  |              /Mul_3             |        Mul        |                   /scale_activation_3/HardSigmoid_output_0, /block.1.2_9/Mul_output_0                   |             /Mul_3_output_0              |                                               |
    |  142  |        /block.3.0_3/Conv        |        Conv       |                             /Mul_3_output_0, onnx::Conv_620, onnx::Conv_621                             |        /block.3.0_3/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  143  |        /block.0.0_11/Conv       |        Conv       |                        /block.3.0_3/Conv_output_0, onnx::Conv_623, onnx::Conv_624                       |       /block.0.0_11/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  144  |    /block.0.2_11/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_11/Conv_output_0                                       |    /block.0.2_11/HardSigmoid_output_0    |                     alpha                     |
    |  145  |        /block.0.2_11/Mul        |        Mul        |                     /block.0.0_11/Conv_output_0, /block.0.2_11/HardSigmoid_output_0                     |        /block.0.2_11/Mul_output_0        |                                               |
    |  146  |        /block.1.0_11/Conv       |        Conv       |                        /block.0.2_11/Mul_output_0, onnx::Conv_626, onnx::Conv_627                       |       /block.1.0_11/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  147  |    /block.1.2_10/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_11/Conv_output_0                                       |    /block.1.2_10/HardSigmoid_output_0    |                     alpha                     |
    |  148  |        /block.1.2_10/Mul        |        Mul        |                     /block.1.0_11/Conv_output_0, /block.1.2_10/HardSigmoid_output_0                     |        /block.1.2_10/Mul_output_0        |                                               |
    |  149  |   /avgpool_4/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_10/Mul_output_0                                       |  /avgpool_4/GlobalAveragePool_output_0   |                                               |
    |  150  |           /fc1_4/Conv           |        Conv       | /avgpool_4/GlobalAveragePool_output_0, 1.features.12.block.2.fc1.weight, 1.features.12.block.2.fc1.bias |           /fc1_4/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  151  |        /activation_4/Relu       |        Relu       |                                           /fc1_4/Conv_output_0                                          |       /activation_4/Relu_output_0        |                                               |
    |  152  |           /fc2_4/Conv           |        Conv       |      /activation_4/Relu_output_0, 1.features.12.block.2.fc2.weight, 1.features.12.block.2.fc2.bias      |           /fc2_4/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  153  | /scale_activation_4/HardSigmoid |    HardSigmoid    |                                           /fc2_4/Conv_output_0                                          | /scale_activation_4/HardSigmoid_output_0 |                     alpha                     |
    |  154  |              /Mul_4             |        Mul        |                   /scale_activation_4/HardSigmoid_output_0, /block.1.2_10/Mul_output_0                  |             /Mul_4_output_0              |                                               |
    |  155  |        /block.3.0_4/Conv        |        Conv       |                             /Mul_4_output_0, onnx::Conv_629, onnx::Conv_630                             |        /block.3.0_4/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  156  |              /Add_7             |        Add        |                          /block.3.0_4/Conv_output_0, /block.3.0_3/Conv_output_0                         |             /Add_7_output_0              |                                               |
    |  157  |        /block.0.0_12/Conv       |        Conv       |                             /Add_7_output_0, onnx::Conv_632, onnx::Conv_633                             |       /block.0.0_12/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  158  |    /block.0.2_12/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_12/Conv_output_0                                       |    /block.0.2_12/HardSigmoid_output_0    |                     alpha                     |
    |  159  |        /block.0.2_12/Mul        |        Mul        |                     /block.0.0_12/Conv_output_0, /block.0.2_12/HardSigmoid_output_0                     |        /block.0.2_12/Mul_output_0        |                                               |
    |  160  |        /block.1.0_12/Conv       |        Conv       |                        /block.0.2_12/Mul_output_0, onnx::Conv_635, onnx::Conv_636                       |       /block.1.0_12/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  161  |    /block.1.2_11/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_12/Conv_output_0                                       |    /block.1.2_11/HardSigmoid_output_0    |                     alpha                     |
    |  162  |        /block.1.2_11/Mul        |        Mul        |                     /block.1.0_12/Conv_output_0, /block.1.2_11/HardSigmoid_output_0                     |        /block.1.2_11/Mul_output_0        |                                               |
    |  163  |   /avgpool_5/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_11/Mul_output_0                                       |  /avgpool_5/GlobalAveragePool_output_0   |                                               |
    |  164  |           /fc1_5/Conv           |        Conv       | /avgpool_5/GlobalAveragePool_output_0, 1.features.13.block.2.fc1.weight, 1.features.13.block.2.fc1.bias |           /fc1_5/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  165  |        /activation_5/Relu       |        Relu       |                                           /fc1_5/Conv_output_0                                          |       /activation_5/Relu_output_0        |                                               |
    |  166  |           /fc2_5/Conv           |        Conv       |      /activation_5/Relu_output_0, 1.features.13.block.2.fc2.weight, 1.features.13.block.2.fc2.bias      |           /fc2_5/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  167  | /scale_activation_5/HardSigmoid |    HardSigmoid    |                                           /fc2_5/Conv_output_0                                          | /scale_activation_5/HardSigmoid_output_0 |                     alpha                     |
    |  168  |              /Mul_5             |        Mul        |                   /scale_activation_5/HardSigmoid_output_0, /block.1.2_11/Mul_output_0                  |             /Mul_5_output_0              |                                               |
    |  169  |        /block.3.0_5/Conv        |        Conv       |                             /Mul_5_output_0, onnx::Conv_638, onnx::Conv_639                             |        /block.3.0_5/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  170  |        /block.0.0_13/Conv       |        Conv       |                        /block.3.0_5/Conv_output_0, onnx::Conv_641, onnx::Conv_642                       |       /block.0.0_13/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  171  |    /block.0.2_13/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_13/Conv_output_0                                       |    /block.0.2_13/HardSigmoid_output_0    |                     alpha                     |
    |  172  |        /block.0.2_13/Mul        |        Mul        |                     /block.0.0_13/Conv_output_0, /block.0.2_13/HardSigmoid_output_0                     |        /block.0.2_13/Mul_output_0        |                                               |
    |  173  |        /block.1.0_13/Conv       |        Conv       |                        /block.0.2_13/Mul_output_0, onnx::Conv_644, onnx::Conv_645                       |       /block.1.0_13/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  174  |    /block.1.2_12/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_13/Conv_output_0                                       |    /block.1.2_12/HardSigmoid_output_0    |                     alpha                     |
    |  175  |        /block.1.2_12/Mul        |        Mul        |                     /block.1.0_13/Conv_output_0, /block.1.2_12/HardSigmoid_output_0                     |        /block.1.2_12/Mul_output_0        |                                               |
    |  176  |   /avgpool_6/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_12/Mul_output_0                                       |  /avgpool_6/GlobalAveragePool_output_0   |                                               |
    |  177  |           /fc1_6/Conv           |        Conv       | /avgpool_6/GlobalAveragePool_output_0, 1.features.14.block.2.fc1.weight, 1.features.14.block.2.fc1.bias |           /fc1_6/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  178  |        /activation_6/Relu       |        Relu       |                                           /fc1_6/Conv_output_0                                          |       /activation_6/Relu_output_0        |                                               |
    |  179  |           /fc2_6/Conv           |        Conv       |      /activation_6/Relu_output_0, 1.features.14.block.2.fc2.weight, 1.features.14.block.2.fc2.bias      |           /fc2_6/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  180  | /scale_activation_6/HardSigmoid |    HardSigmoid    |                                           /fc2_6/Conv_output_0                                          | /scale_activation_6/HardSigmoid_output_0 |                     alpha                     |
    |  181  |              /Mul_6             |        Mul        |                   /scale_activation_6/HardSigmoid_output_0, /block.1.2_12/Mul_output_0                  |             /Mul_6_output_0              |                                               |
    |  182  |        /block.3.0_6/Conv        |        Conv       |                             /Mul_6_output_0, onnx::Conv_647, onnx::Conv_648                             |        /block.3.0_6/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  183  |              /Add_8             |        Add        |                          /block.3.0_6/Conv_output_0, /block.3.0_5/Conv_output_0                         |             /Add_8_output_0              |                                               |
    |  184  |        /block.0.0_14/Conv       |        Conv       |                             /Add_8_output_0, onnx::Conv_650, onnx::Conv_651                             |       /block.0.0_14/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  185  |    /block.0.2_14/HardSigmoid    |    HardSigmoid    |                                       /block.0.0_14/Conv_output_0                                       |    /block.0.2_14/HardSigmoid_output_0    |                     alpha                     |
    |  186  |        /block.0.2_14/Mul        |        Mul        |                     /block.0.0_14/Conv_output_0, /block.0.2_14/HardSigmoid_output_0                     |        /block.0.2_14/Mul_output_0        |                                               |
    |  187  |        /block.1.0_14/Conv       |        Conv       |                        /block.0.2_14/Mul_output_0, onnx::Conv_653, onnx::Conv_654                       |       /block.1.0_14/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  188  |    /block.1.2_13/HardSigmoid    |    HardSigmoid    |                                       /block.1.0_14/Conv_output_0                                       |    /block.1.2_13/HardSigmoid_output_0    |                     alpha                     |
    |  189  |        /block.1.2_13/Mul        |        Mul        |                     /block.1.0_14/Conv_output_0, /block.1.2_13/HardSigmoid_output_0                     |        /block.1.2_13/Mul_output_0        |                                               |
    |  190  |   /avgpool_7/GlobalAveragePool  | GlobalAveragePool |                                        /block.1.2_13/Mul_output_0                                       |  /avgpool_7/GlobalAveragePool_output_0   |                                               |
    |  191  |           /fc1_7/Conv           |        Conv       | /avgpool_7/GlobalAveragePool_output_0, 1.features.15.block.2.fc1.weight, 1.features.15.block.2.fc1.bias |           /fc1_7/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  192  |        /activation_7/Relu       |        Relu       |                                           /fc1_7/Conv_output_0                                          |       /activation_7/Relu_output_0        |                                               |
    |  193  |           /fc2_7/Conv           |        Conv       |      /activation_7/Relu_output_0, 1.features.15.block.2.fc2.weight, 1.features.15.block.2.fc2.bias      |           /fc2_7/Conv_output_0           | dilations, group, kernel_shape, pads, strides |
    |  194  | /scale_activation_7/HardSigmoid |    HardSigmoid    |                                           /fc2_7/Conv_output_0                                          | /scale_activation_7/HardSigmoid_output_0 |                     alpha                     |
    |  195  |              /Mul_7             |        Mul        |                   /scale_activation_7/HardSigmoid_output_0, /block.1.2_13/Mul_output_0                  |             /Mul_7_output_0              |                                               |
    |  196  |        /block.3.0_7/Conv        |        Conv       |                             /Mul_7_output_0, onnx::Conv_656, onnx::Conv_657                             |        /block.3.0_7/Conv_output_0        | dilations, group, kernel_shape, pads, strides |
    |  197  |              /Add_9             |        Add        |                               /block.3.0_7/Conv_output_0, /Add_8_output_0                               |             /Add_9_output_0              |                                               |
    |  198  |       /features.16.0/Conv       |        Conv       |                             /Add_9_output_0, onnx::Conv_659, onnx::Conv_660                             |       /features.16.0/Conv_output_0       | dilations, group, kernel_shape, pads, strides |
    |  199  |    /features.16.2/HardSigmoid   |    HardSigmoid    |                                       /features.16.0/Conv_output_0                                      |   /features.16.2/HardSigmoid_output_0    |                     alpha                     |
    |  200  |        /features.16.2/Mul       |        Mul        |                    /features.16.0/Conv_output_0, /features.16.2/HardSigmoid_output_0                    |       /features.16.2/Mul_output_0        |                                               |
    |  201  |   /avgpool_8/GlobalAveragePool  | GlobalAveragePool |                                       /features.16.2/Mul_output_0                                       |  /avgpool_8/GlobalAveragePool_output_0   |                                               |
    |  202  |             /Flatten            |      Flatten      |                                  /avgpool_8/GlobalAveragePool_output_0                                  |            /Flatten_output_0             |                      axis                     |
    |  203  |        /classifier.0/Gemm       |        Gemm       |                      /Flatten_output_0, 1.classifier.0.weight, 1.classifier.0.bias                      |       /classifier.0/Gemm_output_0        |              alpha, beta, transB              |
    |  204  |    /classifier.1/HardSigmoid    |    HardSigmoid    |                                       /classifier.0/Gemm_output_0                                       |    /classifier.1/HardSigmoid_output_0    |                     alpha                     |
    |  205  |        /classifier.1/Mul        |        Mul        |                     /classifier.0/Gemm_output_0, /classifier.1/HardSigmoid_output_0                     |        /classifier.1/Mul_output_0        |                                               |
    |  206  |        /classifier.3/Gemm       |        Gemm       |                  /classifier.1/Mul_output_0, 1.classifier.3.weight, 1.classifier.3.bias                 |                   522                    |              alpha, beta, transB              |
    +-------+---------------------------------+-------------------+---------------------------------------------------------------------------------------------------------+------------------------------------------+-----------------------------------------------+
    [32mINFO    [0m [34mQuantizing model using static quantization with calibration...[0m
    I0327 21:20:45.625436 140179936118592 quantize.py:54] Quantizing model using static quantization with calibration...
    [32mINFO    [0m [34mUsing CUDA as ONNX execution provider.[0m
    I0327 21:20:45.625665 140179936118592 utils.py:8] Using CUDA as ONNX execution provider.
    [32mINFO    [0m [34mQuantization complete. Model is now calibrated and statically quantized.[0m
    I0327 21:23:01.993722 140179936118592 quantize.py:92] Quantization complete. Model is now calibrated and statically quantized.
    [32mINFO    [0m [34mQuantizing model using dynamic quantization...[0m
    I0327 21:23:02.003372 140179936118592 quantize.py:33] Quantizing model using dynamic quantization...
    [32mINFO    [0m [34mQuantization complete. Model is now dynamically quantized.[0m
    I0327 21:23:03.345017 140179936118592 quantize.py:48] Quantization complete. Model is now dynamically quantized.
    [32mINFO    [0m [34mQuantizing model using automatic mixed precision quantization...[0m
    I0327 21:23:03.345549 140179936118592 quantize.py:98] Quantizing model using automatic mixed precision quantization...
    Adding missing dtypes for 0 outputs
    ['/0/Conv', '/features.0.0/Conv', '/features.0.2/HardSigmoid', '/features.0.2/Mul', '/block.0.0/Conv', '/block.0.2/Relu', '/block.1.0/Conv', '/Add', '/block.0.0_1/Conv', '/block.0.2_1/Relu', '/block.1.0_1/Conv', '/block.1.2/Relu', '/block.2.0/Conv', '/block.0.0_2/Conv', '/block.0.2_2/Relu', '/block.1.0_2/Conv', '/block.1.2_1/Relu', '/block.2.0_1/Conv', '/Add_1', '/block.0.0_3/Conv', '/block.0.2_3/Relu', '/block.1.0_3/Conv', '/block.1.2_2/Relu', '/avgpool/GlobalAveragePool', '/fc1/Conv', '/activation/Relu', '/fc2/Conv', '/scale_activation/HardSigmoid', '/Mul', '/block.3.0/Conv', '/block.0.0_4/Conv', '/block.0.2_4/Relu', '/block.1.0_4/Conv', '/block.1.2_3/Relu', '/avgpool_1/GlobalAveragePool', '/fc1_1/Conv', '/activation_1/Relu', '/fc2_1/Conv', '/scale_activation_1/HardSigmoid', '/Mul_1', '/block.3.0_1/Conv', '/Add_2', '/block.0.0_5/Conv', '/block.0.2_5/Relu', '/block.1.0_5/Conv', '/block.1.2_4/Relu', '/avgpool_2/GlobalAveragePool', '/fc1_2/Conv', '/activation_2/Relu', '/fc2_2/Conv', '/scale_activation_2/HardSigmoid', '/Mul_2', '/block.3.0_2/Conv', '/Add_3', '/block.0.0_6/Conv', '/block.0.2_6/HardSigmoid', '/block.0.2_6/Mul', '/block.1.0_6/Conv', '/block.1.2_5/HardSigmoid', '/block.1.2_5/Mul', '/block.2.0_2/Conv', '/block.0.0_7/Conv', '/block.0.2_7/HardSigmoid', '/block.0.2_7/Mul', '/block.1.0_7/Conv', '/block.1.2_6/HardSigmoid', '/block.1.2_6/Mul', '/block.2.0_3/Conv', '/Add_4', '/block.0.0_8/Conv', '/block.0.2_8/HardSigmoid', '/block.0.2_8/Mul', '/block.1.0_8/Conv', '/block.1.2_7/HardSigmoid', '/block.1.2_7/Mul', '/block.2.0_4/Conv', '/Add_5', '/block.0.0_9/Conv', '/block.0.2_9/HardSigmoid', '/block.0.2_9/Mul', '/block.1.0_9/Conv', '/block.1.2_8/HardSigmoid', '/block.1.2_8/Mul', '/block.2.0_5/Conv', '/Add_6', '/block.0.0_10/Conv', '/block.0.2_10/HardSigmoid', '/block.0.2_10/Mul', '/block.1.0_10/Conv', '/block.1.2_9/HardSigmoid', '/block.1.2_9/Mul', '/avgpool_3/GlobalAveragePool', '/fc1_3/Conv', '/activation_3/Relu', '/fc2_3/Conv', '/scale_activation_3/HardSigmoid', '/Mul_3', '/block.3.0_3/Conv', '/block.0.0_11/Conv', '/block.0.2_11/HardSigmoid', '/block.0.2_11/Mul', '/block.1.0_11/Conv', '/block.1.2_10/HardSigmoid', '/block.1.2_10/Mul', '/avgpool_4/GlobalAveragePool', '/fc1_4/Conv', '/activation_4/Relu', '/fc2_4/Conv', '/scale_activation_4/HardSigmoid', '/Mul_4', '/block.3.0_4/Conv', '/Add_7', '/block.0.0_12/Conv', '/block.0.2_12/HardSigmoid', '/block.0.2_12/Mul', '/block.1.0_12/Conv', '/block.1.2_11/HardSigmoid', '/block.1.2_11/Mul', '/avgpool_5/GlobalAveragePool', '/fc1_5/Conv', '/activation_5/Relu', '/fc2_5/Conv', '/scale_activation_5/HardSigmoid', '/Mul_5', '/block.3.0_5/Conv', '/block.0.0_13/Conv', '/block.0.2_13/HardSigmoid', '/block.0.2_13/Mul', '/block.1.0_13/Conv', '/block.1.2_12/HardSigmoid', '/block.1.2_12/Mul', '/avgpool_6/GlobalAveragePool', '/fc1_6/Conv', '/activation_6/Relu', '/fc2_6/Conv', '/scale_activation_6/HardSigmoid', '/Mul_6', '/block.3.0_6/Conv', '/Add_8', '/block.0.0_14/Conv', '/block.0.2_14/HardSigmoid', '/block.0.2_14/Mul', '/block.1.0_14/Conv', '/block.1.2_13/HardSigmoid', '/block.1.2_13/Mul', '/avgpool_7/GlobalAveragePool', '/fc1_7/Conv', '/activation_7/Relu', '/fc2_7/Conv', '/scale_activation_7/HardSigmoid', '/Mul_7', '/block.3.0_7/Conv', '/Add_9', '/features.16.0/Conv', '/features.16.2/HardSigmoid', '/features.16.2/Mul', '/avgpool_8/GlobalAveragePool', '/Flatten', '/classifier.0/Gemm', '/classifier.1/HardSigmoid', '/classifier.1/Mul', '/classifier.3/Gemm']
    True
    Sanity checks passed. Starting autoconvert.
    Running attempt 1 excluding conversion of 0 nodes
    []
    True
    Attempt succeeded.
    [*162*]
    Done: []
    []
    Final model validated successfully.
    [32mINFO    [0m [34mQuantization complete. Model is now quantized using automatic mixed precision.[0m
    I0327 21:23:15.115900 140179936118592 quantize.py:120] Quantization complete. Model is now quantized using automatic mixed precision.
    [32mINFO    [0m [34mPerforming runtime analysis on original graph...[0m
    I0327 21:23:15.124848 140179936118592 transform.py:170] Performing runtime analysis on original graph...
    [32mINFO    [0m [34mStarting transformation analysis on mobilenetv3_large[0m
    I0327 21:23:15.125154 140179936118592 analysis.py:276] Starting transformation analysis on mobilenetv3_large
    [32mINFO    [0m [34m
    Results mobilenetv3_large:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.10706   |
    |      Average Precision       |   0.10244   |
    |        Average Recall        |   0.11135   |
    |       Average F1 Score       |   0.10068   |
    |         Average Loss         |   2.3038    |
    |       Average Latency        |  17.505 ms  |
    |   Average GPU Power Usage    |  26.257 W   |
    | Inference Energy Consumption | 0.12767 mWh |
    +------------------------------+-------------+[0m
    I0327 21:23:21.650321 140179936118592 analysis.py:404] 
    Results mobilenetv3_large:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.10706   |
    |      Average Precision       |   0.10244   |
    |        Average Recall        |   0.11135   |
    |       Average F1 Score       |   0.10068   |
    |         Average Loss         |   2.3038    |
    |       Average Latency        |  17.505 ms  |
    |   Average GPU Power Usage    |  26.257 W   |
    | Inference Energy Consumption | 0.12767 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/mase_graph/version_1/model.json[0m
    I0327 21:23:21.651654 140179936118592 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/mase_graph/version_1/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on onnx-optimized graph...[0m
    I0327 21:23:21.651822 140179936118592 transform.py:176] Performing runtime analysis on onnx-optimized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0327 21:23:21.652008 140179936118592 analysis.py:65] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [32mINFO    [0m [34mStarting transformation analysis on mobilenetv3_large-onnx[0m
    I0327 21:23:21.885762 140179936118592 analysis.py:276] Starting transformation analysis on mobilenetv3_large-onnx
    [32mINFO    [0m [34m
    Results mobilenetv3_large-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.091344   |
    |      Average Precision       |   0.063765   |
    |        Average Recall        |   0.092105   |
    |       Average F1 Score       |   0.058955   |
    |         Average Loss         |    2.3026    |
    |       Average Latency        |  3.6202 ms   |
    |   Average GPU Power Usage    |   48.878 W   |
    | Inference Energy Consumption | 0.049152 mWh |
    +------------------------------+--------------+[0m
    I0327 21:23:27.055771 140179936118592 analysis.py:404] 
    Results mobilenetv3_large-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.091344   |
    |      Average Precision       |   0.063765   |
    |        Average Recall        |   0.092105   |
    |       Average F1 Score       |   0.058955   |
    |         Average Loss         |    2.3026    |
    |       Average Latency        |  3.6202 ms   |
    |   Average GPU Power Usage    |   48.878 W   |
    | Inference Energy Consumption | 0.049152 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_4/model.json[0m
    I0327 21:23:27.058081 140179936118592 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_4/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on static quantized graph...[0m
    I0327 21:23:27.067597 140179936118592 transform.py:191] Performing runtime analysis on static quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0327 21:23:27.068019 140179936118592 analysis.py:65] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [0;93m2024-03-27 21:23:27.431358608 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 65 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.[m
    [0;93m2024-03-27 21:23:27.436990877 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
    [0;93m2024-03-27 21:23:27.437005349 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
    [32mINFO    [0m [34mStarting transformation analysis on mobilenetv3_large-onnx[0m
    I0327 21:23:27.465283 140179936118592 analysis.py:276] Starting transformation analysis on mobilenetv3_large-onnx
    [32mINFO    [0m [34m
    Results mobilenetv3_large-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.094741   |
    |      Average Precision       |   0.060424   |
    |        Average Recall        |   0.095888   |
    |       Average F1 Score       |   0.056649   |
    |         Average Loss         |    2.3026    |
    |       Average Latency        |  7.3654 ms   |
    |   Average GPU Power Usage    |   47.643 W   |
    | Inference Energy Consumption | 0.097475 mWh |
    +------------------------------+--------------+[0m
    I0327 21:23:33.635719 140179936118592 analysis.py:404] 
    Results mobilenetv3_large-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.094741   |
    |      Average Precision       |   0.060424   |
    |        Average Recall        |   0.095888   |
    |       Average F1 Score       |   0.056649   |
    |         Average Loss         |    2.3026    |
    |       Average Latency        |  7.3654 ms   |
    |   Average GPU Power Usage    |   47.643 W   |
    | Inference Energy Consumption | 0.097475 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_5/model.json[0m
    I0327 21:23:33.638137 140179936118592 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_5/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on dynamic quantized graph...[0m
    I0327 21:23:33.652758 140179936118592 transform.py:196] Performing runtime analysis on dynamic quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0327 21:23:33.655529 140179936118592 analysis.py:65] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [0;93m2024-03-27 21:23:33.926589413 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 194 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.[m
    [0;93m2024-03-27 21:23:33.932506394 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
    [0;93m2024-03-27 21:23:33.932521143 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
    [32mINFO    [0m [34mStarting transformation analysis on mobilenetv3_large-onnx[0m
    I0327 21:23:33.955991 140179936118592 analysis.py:276] Starting transformation analysis on mobilenetv3_large-onnx
    [32mINFO    [0m [34m
    Results mobilenetv3_large-onnx:
    +------------------------------+------------+
    |      Metric (Per Batch)      |   Value    |
    +------------------------------+------------+
    |    Average Test Accuracy     |  0.095697  |
    |      Average Precision       |  0.06046   |
    |        Average Recall        |  0.096711  |
    |       Average F1 Score       |  0.058563  |
    |         Average Loss         |   2.3026   |
    |       Average Latency        | 838.45 ms  |
    |   Average GPU Power Usage    |  23.897 W  |
    | Inference Energy Consumption | 5.5656 mWh |
    +------------------------------+------------+[0m
    I0327 21:25:04.969060 140179936118592 analysis.py:404] 
    Results mobilenetv3_large-onnx:
    +------------------------------+------------+
    |      Metric (Per Batch)      |   Value    |
    +------------------------------+------------+
    |    Average Test Accuracy     |  0.095697  |
    |      Average Precision       |  0.06046   |
    |        Average Recall        |  0.096711  |
    |       Average F1 Score       |  0.058563  |
    |         Average Loss         |   2.3026   |
    |       Average Latency        | 838.45 ms  |
    |   Average GPU Power Usage    |  23.897 W  |
    | Inference Energy Consumption | 5.5656 mWh |
    +------------------------------+------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_6/model.json[0m
    I0327 21:25:04.971395 140179936118592 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_6/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on auto mixed precision quantized graph...[0m
    I0327 21:25:04.983319 140179936118592 transform.py:201] Performing runtime analysis on auto mixed precision quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0327 21:25:04.983848 140179936118592 analysis.py:65] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [32mINFO    [0m [34mStarting transformation analysis on mobilenetv3_large-onnx[0m
    I0327 21:25:05.165926 140179936118592 analysis.py:276] Starting transformation analysis on mobilenetv3_large-onnx
    [32mINFO    [0m [34m
    Results mobilenetv3_large-onnx:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.10023   |
    |      Average Precision       |  0.0090688  |
    |        Average Recall        |   0.09523   |
    |       Average F1 Score       |  0.016561   |
    |         Average Loss         |   2.3026    |
    |       Average Latency        |  11.428 ms  |
    |   Average GPU Power Usage    |  34.991 W   |
    | Inference Energy Consumption | 0.11107 mWh |
    +------------------------------+-------------+[0m
    I0327 21:25:15.329111 140179936118592 analysis.py:404] 
    Results mobilenetv3_large-onnx:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.10023   |
    |      Average Precision       |  0.0090688  |
    |        Average Recall        |   0.09523   |
    |       Average F1 Score       |  0.016561   |
    |         Average Loss         |   2.3026    |
    |       Average Latency        |  11.428 ms  |
    |   Average GPU Power Usage    |  34.991 W   |
    | Inference Energy Consumption | 0.11107 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_7/model.json[0m
    I0327 21:25:15.331540 140179936118592 analysis.py:90] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/mobilenetv3_large_cls_mnist_2024-03-27/onnx/version_7/model.json
    [32mINFO    [0m [34mSaved mase graph to /root/mase/mase_output/mobilenetv3_large_cls_mnist_2024-03-27/software/transform/transformed_ckpt[0m
    I0327 21:25:15.531451 140179936118592 save_and_load.py:147] Saved mase graph to /root/mase/mase_output/mobilenetv3_large_cls_mnist_2024-03-27/software/transform/transformed_ckpt
    [32mINFO    [0m [34mTransformation is completed[0m
    I0327 21:25:15.531878 140179936118592 cli.py:383] Transformation is completed

