# Advanced: ONNX Runtime Tutorial

This notebook is designed to demonstrate the features of the ONNXRT passes integrated into MASE as part of the MASERT framework. The following demonstrations were run on a NVIDIA RTX A2000 GPU with a Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz CPU.

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
    onnx_runtime_interface_pass,
    )

set_logging_verbosity("info")
```

    [2024-03-29 13:46:19,035] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)


    [32mINFO    [0m [34mSet logging level to info[0m
    WARNING: Logging before flag parsing goes to stderr.
    I0329 13:46:21.531338 140128553666368 logger.py:44] Set logging level to info


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
!ch train --config {JSC_TOML_PATH} --accelerator gpu
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

    [2024-03-28 23:09:44,122] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0328 23:09:47.151937 140014036379456 seed.py:54] Seed set to 0
    +-------------------------+------------------------+--------------------------+--------------------------+--------------------------+
    | Name                    |        Default         |       Config. File       |     Manual Override      |        Effective         |
    +-------------------------+------------------------+--------------------------+--------------------------+--------------------------+
    | task                    |     [38;5;8mclassification[0m     |           cls            |                          |           cls            |
    | load_name               |          [38;5;8mNone[0m          | [38;5;8m../mase_output/vgg7-pre-[0m | /root/mase/mase_output/v | /root/mase/mase_output/v |
    |                         |                        |      [38;5;8mtrained/test-[0m       |  gg7-pre-trained/test-   |  gg7-pre-trained/test-   |
    |                         |                        |     [38;5;8maccu-0.9332.ckpt[0m     |     accu-0.9332.ckpt     |     accu-0.9332.ckpt     |
    | load_type               |           [38;5;8mmz[0m           |            [38;5;8mpl[0m            |            pl            |            pl            |
    | batch_size              |          [38;5;8m128[0m           |            64            |                          |            64            |
    | to_debug                |         False          |                          |                          |          False           |
    | log_level               |          info          |                          |                          |           info           |
    | report_to               |      tensorboard       |                          |                          |       tensorboard        |
    | seed                    |           0            |                          |                          |            0             |
    | quant_config            |          None          |                          |                          |           None           |
    | training_optimizer      |          adam          |                          |                          |           adam           |
    | trainer_precision       |        16-mixed        |                          |                          |         16-mixed         |
    | learning_rate           |         [38;5;8m1e-05[0m          |          0.001           |                          |          0.001           |
    | weight_decay            |           0            |                          |                          |            0             |
    | max_epochs              |           [38;5;8m20[0m           |            10            |                          |            10            |
    | max_steps               |           -1           |                          |                          |            -1            |
    | accumulate_grad_batches |           1            |                          |                          |            1             |
    | log_every_n_steps       |           50           |                          |                          |            50            |
    | num_workers             |           28           |                          |                          |            28            |
    | num_devices             |           1            |                          |                          |            1             |
    | num_nodes               |           1            |                          |                          |            1             |
    | accelerator             |          [38;5;8mauto[0m          |           gpu            |                          |           gpu            |
    | strategy                |          auto          |                          |                          |           auto           |
    | is_to_auto_requeue      |         False          |                          |                          |          False           |
    | github_ci               |         False          |                          |                          |          False           |
    | disable_dataset_cache   |         False          |                          |                          |          False           |
    | target                  |  xcu250-figd2104-2L-e  |                          |                          |   xcu250-figd2104-2L-e   |
    | num_targets             |          100           |                          |                          |           100            |
    | is_pretrained           |         False          |                          |                          |          False           |
    | max_token_len           |          512           |                          |                          |           512            |
    | project_dir             | /root/mase/mase_output |                          |                          |  /root/mase/mase_output  |
    | project                 |          None          |                          |                          |           None           |
    | model                   |          [38;5;8mNone[0m          |           vgg7           |                          |           vgg7           |
    | dataset                 |          [38;5;8mNone[0m          |         cifar10          |                          |         cifar10          |
    | t_max                   |           20           |                          |                          |            20            |
    | eta_min                 |         1e-06          |                          |                          |          1e-06           |
    +-------------------------+------------------------+--------------------------+--------------------------+--------------------------+
    [32mINFO    [0m [34mInitialising model 'vgg7'...[0m
    I0328 23:09:47.162513 140014036379456 cli.py:846] Initialising model 'vgg7'...
    [32mINFO    [0m [34mInitialising dataset 'cifar10'...[0m
    I0328 23:09:47.270153 140014036379456 cli.py:874] Initialising dataset 'cifar10'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-28[0m
    I0328 23:09:47.270543 140014036379456 cli.py:910] Project will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-28
    [32mINFO    [0m [34mTransforming model 'vgg7'...[0m
    I0328 23:09:47.398360 140014036379456 cli.py:370] Transforming model 'vgg7'...
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0328 23:09:53.493722 140014036379456 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0328 23:09:53.612564 140014036379456 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0328 23:10:35.118083 140014036379456 onnx_runtime.py:88] Converting PyTorch model to ONNX...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-28[0m
    I0328 23:10:35.119032 140014036379456 onnx_runtime.py:90] Project will be created at /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-28
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-28/optimized/version_3/model.onnx[0m
    I0328 23:10:43.779212 140014036379456 onnx_runtime.py:108] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-28/optimized/version_3/model.onnx
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
    I0328 23:10:43.897069 140014036379456 onnx_runtime.py:146] ONNX Model Summary: 
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
    [32mINFO    [0m [34mQuantizing model using static quantization with calibration...[0m
    I0328 23:10:44.711065 140014036379456 quantize.py:68] Quantizing model using static quantization with calibration...
    [32mINFO    [0m [34mUsing CUDA as ONNX execution provider.[0m
    I0328 23:10:44.711297 140014036379456 utils.py:9] Using CUDA as ONNX execution provider.
    [32mINFO    [0m [34mQuantization complete. Model is now calibrated and statically quantized.[0m
    I0328 23:15:18.611725 140014036379456 quantize.py:122] Quantization complete. Model is now calibrated and statically quantized.
    [32mINFO    [0m [34mQuantizing model using dynamic quantization...[0m
    I0328 23:15:18.623580 140014036379456 quantize.py:45] Quantizing model using dynamic quantization...
    [32mINFO    [0m [34mQuantization complete. Model is now dynamically quantized.[0m
    I0328 23:15:19.266244 140014036379456 quantize.py:62] Quantization complete. Model is now dynamically quantized.
    [32mINFO    [0m [34mQuantizing model using automatic mixed precision quantization...[0m
    I0328 23:15:19.266812 140014036379456 quantize.py:132] Quantizing model using automatic mixed precision quantization...
    Adding missing dtypes for 0 outputs
    ['/feature_layers.0/Conv', '/feature_layers.2/Relu', '/feature_layers.3/Conv', '/feature_layers.5/Relu', '/feature_layers.6/MaxPool', '/feature_layers.7/Conv', '/feature_layers.9/Relu', '/feature_layers.10/Conv', '/feature_layers.12/Relu', '/feature_layers.13/MaxPool', '/feature_layers.14/Conv', '/feature_layers.16/Relu', '/feature_layers.17/Conv', '/feature_layers.19/Relu', '/feature_layers.20/MaxPool', '/Reshape', '/classifier.0/Gemm', '/classifier.1/Relu', '/classifier.2/Gemm', '/classifier.3/Relu', '/last_layer/Gemm']
    True
    Sanity checks passed. Starting autoconvert.
    Running attempt 1 excluding conversion of 0 nodes
    []
    True
    Attempt succeeded.
    [*21*]
    Done: []
    []
    Final model validated successfully.
    [32mINFO    [0m [34mQuantization complete. Model is now quantized using automatic mixed precision.[0m
    I0328 23:15:33.080978 140014036379456 quantize.py:164] Quantization complete. Model is now quantized using automatic mixed precision.
    [32mINFO    [0m [34mPerforming runtime analysis on original graph...[0m
    I0328 23:15:33.103024 140014036379456 transform.py:257] Performing runtime analysis on original graph...
    [32mINFO    [0m [34mStarting transformation analysis on vgg7[0m
    I0328 23:15:33.103300 140014036379456 runtime_analysis.py:357] Starting transformation analysis on vgg7
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.91831   |
    |      Average Precision       |   0.91791   |
    |        Average Recall        |   0.91793   |
    |       Average F1 Score       |   0.91778   |
    |         Average Loss         |   0.24676   |
    |       Average Latency        |  8.6022 ms  |
    |   Average GPU Power Usage    |  50.352 W   |
    | Inference Energy Consumption | 0.12032 mWh |
    +------------------------------+-------------+[0m
    I0328 23:15:43.756563 140014036379456 runtime_analysis.py:521] 
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.91831   |
    |      Average Precision       |   0.91791   |
    |        Average Recall        |   0.91793   |
    |       Average F1 Score       |   0.91778   |
    |         Average Loss         |   0.24676   |
    |       Average Latency        |  8.6022 ms  |
    |   Average GPU Power Usage    |  50.352 W   |
    | Inference Energy Consumption | 0.12032 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_47/model.json[0m
    I0328 23:15:43.758996 140014036379456 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_47/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on onnx-optimized graph...[0m
    I0328 23:15:43.759325 140014036379456 transform.py:263] Performing runtime analysis on onnx-optimized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0328 23:15:43.759659 140014036379456 runtime_analysis.py:108] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0328 23:15:43.972420 140014036379456 runtime_analysis.py:357] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.93054    |
    |      Average Precision       |   0.93154    |
    |        Average Recall        |   0.93158    |
    |       Average F1 Score       |   0.93138    |
    |         Average Loss         |   0.22327    |
    |       Average Latency        |  6.0678 ms   |
    |   Average GPU Power Usage    |   55.829 W   |
    | Inference Energy Consumption | 0.094099 mWh |
    +------------------------------+--------------+[0m
    I0328 23:15:53.476423 140014036379456 runtime_analysis.py:521] 
    Results vgg7-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.93054    |
    |      Average Precision       |   0.93154    |
    |        Average Recall        |   0.93158    |
    |       Average F1 Score       |   0.93138    |
    |         Average Loss         |   0.22327    |
    |       Average Latency        |  6.0678 ms   |
    |   Average GPU Power Usage    |   55.829 W   |
    | Inference Energy Consumption | 0.094099 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/onnx/version_9/model.json[0m
    I0328 23:15:53.478985 140014036379456 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/onnx/version_9/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on static quantized graph...[0m
    I0328 23:15:53.539209 140014036379456 transform.py:282] Performing runtime analysis on static quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0328 23:15:53.539841 140014036379456 runtime_analysis.py:108] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [0;93m2024-03-28 23:15:53.640481870 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 9 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.[m
    [0;93m2024-03-28 23:15:53.641186374 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
    [0;93m2024-03-28 23:15:53.641201703 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0328 23:15:53.654182 140014036379456 runtime_analysis.py:357] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.93175   |
    |      Average Precision       |   0.93244   |
    |        Average Recall        |   0.93224   |
    |       Average F1 Score       |   0.93212   |
    |         Average Loss         |   0.22437   |
    |       Average Latency        |  3.8347 ms  |
    |   Average GPU Power Usage    |  54.555 W   |
    | Inference Energy Consumption | 0.14388 mWh |
    +------------------------------+-------------+[0m
    I0328 23:16:03.469463 140014036379456 runtime_analysis.py:521] 
    Results vgg7-onnx:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.93175   |
    |      Average Precision       |   0.93244   |
    |        Average Recall        |   0.93224   |
    |       Average F1 Score       |   0.93212   |
    |         Average Loss         |   0.22437   |
    |       Average Latency        |  3.8347 ms  |
    |   Average GPU Power Usage    |  54.555 W   |
    | Inference Energy Consumption | 0.14388 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/onnx/version_10/model.json[0m
    I0328 23:16:03.472026 140014036379456 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/onnx/version_10/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on dynamic quantized graph...[0m
    I0328 23:16:03.488069 140014036379456 transform.py:290] Performing runtime analysis on dynamic quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0328 23:16:03.489004 140014036379456 runtime_analysis.py:108] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [0;93m2024-03-28 23:16:03.624021594 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 26 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.[m
    [0;93m2024-03-28 23:16:03.624989637 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
    [0;93m2024-03-28 23:16:03.625007282 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0328 23:16:03.641838 140014036379456 runtime_analysis.py:357] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+------------+
    |      Metric (Per Batch)      |   Value    |
    +------------------------------+------------+
    |    Average Test Accuracy     |  0.93162   |
    |      Average Precision       |  0.93261   |
    |        Average Recall        |  0.93257   |
    |       Average F1 Score       |  0.93241   |
    |         Average Loss         |  0.22253   |
    |       Average Latency        |  5.1273 ms |
    |   Average GPU Power Usage    |  22.86 W   |
    | Inference Energy Consumption | 0.1748 mWh |
    +------------------------------+------------+[0m
    I0328 23:18:23.964464 140014036379456 runtime_analysis.py:521] 
    Results vgg7-onnx:
    +------------------------------+------------+
    |      Metric (Per Batch)      |   Value    |
    +------------------------------+------------+
    |    Average Test Accuracy     |  0.93162   |
    |      Average Precision       |  0.93261   |
    |        Average Recall        |  0.93257   |
    |       Average F1 Score       |  0.93241   |
    |         Average Loss         |  0.22253   |
    |       Average Latency        |  5.1273 ms |
    |   Average GPU Power Usage    |  22.86 W   |
    | Inference Energy Consumption | 0.1748 mWh |
    +------------------------------+------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/onnx/version_11/model.json[0m
    I0328 23:18:23.966642 140014036379456 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/onnx/version_11/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on auto mixed precision quantized graph...[0m
    I0328 23:18:23.987104 140014036379456 transform.py:298] Performing runtime analysis on auto mixed precision quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0328 23:18:23.987448 140014036379456 runtime_analysis.py:108] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0328 23:18:24.154293 140014036379456 runtime_analysis.py:357] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.93087    |
    |      Average Precision       |   0.93188    |
    |        Average Recall        |   0.93191    |
    |       Average F1 Score       |   0.93172    |
    |         Average Loss         |   0.22324    |
    |       Average Latency        |  5.4846 ms   |
    |   Average GPU Power Usage    |   50.354 W   |
    | Inference Energy Consumption | 0.076714 mWh |
    +------------------------------+--------------+[0m
    I0328 23:18:33.854084 140014036379456 runtime_analysis.py:521] 
    Results vgg7-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.93087    |
    |      Average Precision       |   0.93188    |
    |        Average Recall        |   0.93191    |
    |       Average F1 Score       |   0.93172    |
    |         Average Loss         |   0.22324    |
    |       Average Latency        |  5.4846 ms   |
    |   Average GPU Power Usage    |   50.354 W   |
    | Inference Energy Consumption | 0.076714 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/onnx/version_12/model.json[0m
    I0328 23:18:33.855542 140014036379456 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/onnx/version_12/model.json


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
VGG_TOML_PATH = "../../../machop/configs/onnx/vgg7_gpu_quant.toml"
VGG_CHECKPOINT_PATH = "../../../mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt"
!ch transform --config {VGG_TOML_PATH} --load {VGG_CHECKPOINT_PATH} --load-type pl
```

    [2024-03-29 13:49:26,029] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0329 13:49:29.166126 139783521261376 seed.py:54] Seed set to 0
    +-------------------------+------------------------+--------------------------+--------------------------+--------------------------+
    | Name                    |        Default         |       Config. File       |     Manual Override      |        Effective         |
    +-------------------------+------------------------+--------------------------+--------------------------+--------------------------+
    | task                    |     [38;5;8mclassification[0m     |           cls            |                          |           cls            |
    | load_name               |          [38;5;8mNone[0m          | [38;5;8m../mase_output/vgg7-pre-[0m | /root/mase/mase_output/v | /root/mase/mase_output/v |
    |                         |                        |      [38;5;8mtrained/test-[0m       |  gg7-pre-trained/test-   |  gg7-pre-trained/test-   |
    |                         |                        |     [38;5;8maccu-0.9332.ckpt[0m     |     accu-0.9332.ckpt     |     accu-0.9332.ckpt     |
    | load_type               |           [38;5;8mmz[0m           |            [38;5;8mpl[0m            |            pl            |            pl            |
    | batch_size              |          [38;5;8m128[0m           |            64            |                          |            64            |
    | to_debug                |         False          |                          |                          |          False           |
    | log_level               |          info          |                          |                          |           info           |
    | report_to               |      tensorboard       |                          |                          |       tensorboard        |
    | seed                    |           0            |                          |                          |            0             |
    | quant_config            |          None          |                          |                          |           None           |
    | training_optimizer      |          adam          |                          |                          |           adam           |
    | trainer_precision       |        16-mixed        |                          |                          |         16-mixed         |
    | learning_rate           |         [38;5;8m1e-05[0m          |          0.001           |                          |          0.001           |
    | weight_decay            |           0            |                          |                          |            0             |
    | max_epochs              |           [38;5;8m20[0m           |            10            |                          |            10            |
    | max_steps               |           -1           |                          |                          |            -1            |
    | accumulate_grad_batches |           1            |                          |                          |            1             |
    | log_every_n_steps       |           50           |                          |                          |            50            |
    | num_workers             |           28           |                          |                          |            28            |
    | num_devices             |           1            |                          |                          |            1             |
    | num_nodes               |           1            |                          |                          |            1             |
    | accelerator             |          [38;5;8mauto[0m          |           gpu            |                          |           gpu            |
    | strategy                |          auto          |                          |                          |           auto           |
    | is_to_auto_requeue      |         False          |                          |                          |          False           |
    | github_ci               |         False          |                          |                          |          False           |
    | disable_dataset_cache   |         False          |                          |                          |          False           |
    | target                  |  xcu250-figd2104-2L-e  |                          |                          |   xcu250-figd2104-2L-e   |
    | num_targets             |          100           |                          |                          |           100            |
    | is_pretrained           |         False          |                          |                          |          False           |
    | max_token_len           |          512           |                          |                          |           512            |
    | project_dir             | /root/mase/mase_output |                          |                          |  /root/mase/mase_output  |
    | project                 |          None          |                          |                          |           None           |
    | model                   |          [38;5;8mNone[0m          |           vgg7           |                          |           vgg7           |
    | dataset                 |          [38;5;8mNone[0m          |         cifar10          |                          |         cifar10          |
    | t_max                   |           20           |                          |                          |            20            |
    | eta_min                 |         1e-06          |                          |                          |          1e-06           |
    +-------------------------+------------------------+--------------------------+--------------------------+--------------------------+
    [32mINFO    [0m [34mInitialising model 'vgg7'...[0m
    I0329 13:49:29.176632 139783521261376 cli.py:846] Initialising model 'vgg7'...
    [32mINFO    [0m [34mInitialising dataset 'cifar10'...[0m
    I0329 13:49:29.286156 139783521261376 cli.py:874] Initialising dataset 'cifar10'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-29[0m
    I0329 13:49:29.287228 139783521261376 cli.py:910] Project will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-29
    [32mINFO    [0m [34mTransforming model 'vgg7'...[0m
    I0329 13:49:29.419221 139783521261376 cli.py:370] Transforming model 'vgg7'...
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0329 13:49:35.331030 139783521261376 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0329 13:49:35.445306 139783521261376 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0329 13:50:32.507291 139783521261376 onnx_runtime.py:88] Converting PyTorch model to ONNX...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-29[0m
    I0329 13:50:32.508896 139783521261376 onnx_runtime.py:90] Project will be created at /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-29
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-29/optimized/version_0/model.onnx[0m
    I0329 13:50:53.587861 139783521261376 onnx_runtime.py:108] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/onnxrt/vgg7_cls_cifar10_2024-03-29/optimized/version_0/model.onnx
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
    I0329 13:50:53.719763 139783521261376 onnx_runtime.py:146] ONNX Model Summary: 
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
    [32mINFO    [0m [34mQuantizing model using static quantization with calibration...[0m
    I0329 13:50:54.538058 139783521261376 quantize.py:68] Quantizing model using static quantization with calibration...
    [32mINFO    [0m [34mUsing CUDA as ONNX execution provider.[0m
    I0329 13:50:54.538278 139783521261376 utils.py:9] Using CUDA as ONNX execution provider.
    [32mINFO    [0m [34mQuantization complete. Model is now calibrated and statically quantized.[0m
    I0329 13:55:09.302043 139783521261376 quantize.py:122] Quantization complete. Model is now calibrated and statically quantized.
    [32mINFO    [0m [34mQuantizing model using dynamic quantization...[0m
    I0329 13:55:09.317632 139783521261376 quantize.py:45] Quantizing model using dynamic quantization...
    [32mINFO    [0m [34mQuantization complete. Model is now dynamically quantized.[0m
    I0329 13:55:10.003557 139783521261376 quantize.py:62] Quantization complete. Model is now dynamically quantized.
    [32mINFO    [0m [34mQuantizing model using automatic mixed precision quantization...[0m
    I0329 13:55:10.004111 139783521261376 quantize.py:132] Quantizing model using automatic mixed precision quantization...
    Adding missing dtypes for 0 outputs
    ['/feature_layers.0/Conv', '/feature_layers.2/Relu', '/feature_layers.3/Conv', '/feature_layers.5/Relu', '/feature_layers.6/MaxPool', '/feature_layers.7/Conv', '/feature_layers.9/Relu', '/feature_layers.10/Conv', '/feature_layers.12/Relu', '/feature_layers.13/MaxPool', '/feature_layers.14/Conv', '/feature_layers.16/Relu', '/feature_layers.17/Conv', '/feature_layers.19/Relu', '/feature_layers.20/MaxPool', '/Reshape', '/classifier.0/Gemm', '/classifier.1/Relu', '/classifier.2/Gemm', '/classifier.3/Relu', '/last_layer/Gemm']
    True
    Sanity checks passed. Starting autoconvert.
    Running attempt 1 excluding conversion of 0 nodes
    []
    True
    Attempt succeeded.
    [*21*]
    Done: []
    []
    Final model validated successfully.
    [32mINFO    [0m [34mQuantization complete. Model is now quantized using automatic mixed precision.[0m
    I0329 13:55:35.887633 139783521261376 quantize.py:164] Quantization complete. Model is now quantized using automatic mixed precision.
    [32mINFO    [0m [34mPerforming runtime analysis on original graph...[0m
    I0329 13:55:35.906447 139783521261376 transform.py:257] Performing runtime analysis on original graph...
    [32mINFO    [0m [34mStarting transformation analysis on vgg7[0m
    I0329 13:55:35.906747 139783521261376 runtime_analysis.py:357] Starting transformation analysis on vgg7
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+-----------+
    |      Metric (Per Batch)      |   Value   |
    +------------------------------+-----------+
    |    Average Test Accuracy     |  0.91831  |
    |      Average Precision       |  0.91791  |
    |        Average Recall        |  0.91793  |
    |       Average F1 Score       |  0.91778  |
    |         Average Loss         |  0.24676  |
    |       Average Latency        | 8.5586 ms |
    |   Average GPU Power Usage    | 52.579 W  |
    | Inference Energy Consumption | 0.125 mWh |
    +------------------------------+-----------+[0m
    I0329 13:55:58.685605 139783521261376 runtime_analysis.py:521] 
    Results vgg7:
    +------------------------------+-----------+
    |      Metric (Per Batch)      |   Value   |
    +------------------------------+-----------+
    |    Average Test Accuracy     |  0.91831  |
    |      Average Precision       |  0.91791  |
    |        Average Recall        |  0.91793  |
    |       Average F1 Score       |  0.91778  |
    |         Average Loss         |  0.24676  |
    |       Average Latency        | 8.5586 ms |
    |   Average GPU Power Usage    | 52.579 W  |
    | Inference Energy Consumption | 0.125 mWh |
    +------------------------------+-----------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-29/mase_graph/version_0/model.json[0m
    I0329 13:55:58.687570 139783521261376 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-29/mase_graph/version_0/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on onnx-optimized graph...[0m
    I0329 13:55:58.687825 139783521261376 transform.py:263] Performing runtime analysis on onnx-optimized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0329 13:55:58.688159 139783521261376 runtime_analysis.py:108] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0329 13:55:58.822749 139783521261376 runtime_analysis.py:357] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.93054    |
    |      Average Precision       |   0.93154    |
    |        Average Recall        |   0.93158    |
    |       Average F1 Score       |   0.93138    |
    |         Average Loss         |   0.22327    |
    |       Average Latency        |  6.0628 ms   |
    |   Average GPU Power Usage    |   53.26 W    |
    | Inference Energy Consumption | 0.089695 mWh |
    +------------------------------+--------------+[0m
    I0329 13:56:20.459139 139783521261376 runtime_analysis.py:521] 
    Results vgg7-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.93054    |
    |      Average Precision       |   0.93154    |
    |        Average Recall        |   0.93158    |
    |       Average F1 Score       |   0.93138    |
    |         Average Loss         |   0.22327    |
    |       Average Latency        |  6.0628 ms   |
    |   Average GPU Power Usage    |   53.26 W    |
    | Inference Energy Consumption | 0.089695 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-29/onnx/version_0/model.json[0m
    I0329 13:56:20.461472 139783521261376 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-29/onnx/version_0/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on static quantized graph...[0m
    I0329 13:56:20.475034 139783521261376 transform.py:282] Performing runtime analysis on static quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0329 13:56:20.475347 139783521261376 runtime_analysis.py:108] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [0;93m2024-03-29 13:56:20.543785227 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 9 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.[m
    [0;93m2024-03-29 13:56:20.544404009 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
    [0;93m2024-03-29 13:56:20.544417333 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0329 13:56:20.554211 139783521261376 runtime_analysis.py:357] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.9316    |
    |      Average Precision       |   0.93244   |
    |        Average Recall        |   0.93224   |
    |       Average F1 Score       |   0.93212   |
    |         Average Loss         |   0.22435   |
    |       Average Latency        |  3.3334 ms  |
    |   Average GPU Power Usage    |  58.211 W   |
    | Inference Energy Consumption | 0.14364 mWh |
    +------------------------------+-------------+[0m
    I0329 13:56:42.742136 139783521261376 runtime_analysis.py:521] 
    Results vgg7-onnx:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.9316    |
    |      Average Precision       |   0.93244   |
    |        Average Recall        |   0.93224   |
    |       Average F1 Score       |   0.93212   |
    |         Average Loss         |   0.22435   |
    |       Average Latency        |  3.3334 ms  |
    |   Average GPU Power Usage    |  58.211 W   |
    | Inference Energy Consumption | 0.14364 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-29/onnx/version_1/model.json[0m
    I0329 13:56:42.744704 139783521261376 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-29/onnx/version_1/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on dynamic quantized graph...[0m
    I0329 13:56:42.759592 139783521261376 transform.py:290] Performing runtime analysis on dynamic quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0329 13:56:42.760658 139783521261376 runtime_analysis.py:108] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [0;93m2024-03-29 13:56:42.846650769 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 26 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.[m
    [0;93m2024-03-29 13:56:42.847548047 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
    [0;93m2024-03-29 13:56:42.847563259 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0329 13:56:42.863932 139783521261376 runtime_analysis.py:357] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+------------+
    |      Metric (Per Batch)      |   Value    |
    +------------------------------+------------+
    |    Average Test Accuracy     |  0.93162   |
    |      Average Precision       |  0.93261   |
    |        Average Recall        |  0.93257   |
    |       Average F1 Score       |  0.93241   |
    |         Average Loss         |  0.22253   |
    |       Average Latency        |  5.2453 ms |
    |   Average GPU Power Usage    |  22.924 W  |
    | Inference Energy Consumption | 0.0742 mWh |
    +------------------------------+------------+[0m
    I0329 13:59:46.169317 139783521261376 runtime_analysis.py:521] 
    Results vgg7-onnx:
    +------------------------------+------------+
    |      Metric (Per Batch)      |   Value    |
    +------------------------------+------------+
    |    Average Test Accuracy     |  0.93162   |
    |      Average Precision       |  0.93261   |
    |        Average Recall        |  0.93257   |
    |       Average F1 Score       |  0.93241   |
    |         Average Loss         |  0.22253   |
    |       Average Latency        |  5.2453 ms |
    |   Average GPU Power Usage    |  22.924 W  |
    | Inference Energy Consumption | 0.0742 mWh |
    +------------------------------+------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-29/onnx/version_2/model.json[0m
    I0329 13:59:46.174946 139783521261376 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-29/onnx/version_2/model.json
    [32mINFO    [0m [34mPerforming runtime analysis on auto mixed precision quantized graph...[0m
    I0329 13:59:46.208538 139783521261376 transform.py:298] Performing runtime analysis on auto mixed precision quantized graph...
    [32mINFO    [0m [34mUsing ['CUDAExecutionProvider'] as ONNX execution provider.[0m
    I0329 13:59:46.209225 139783521261376 runtime_analysis.py:108] Using ['CUDAExecutionProvider'] as ONNX execution provider.
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-onnx[0m
    I0329 13:59:46.289226 139783521261376 runtime_analysis.py:357] Starting transformation analysis on vgg7-onnx
    [32mINFO    [0m [34m
    Results vgg7-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.93087    |
    |      Average Precision       |   0.93188    |
    |        Average Recall        |   0.93191    |
    |       Average F1 Score       |   0.93172    |
    |         Average Loss         |   0.22324    |
    |       Average Latency        |  4.9185 ms   |
    |   Average GPU Power Usage    |   49.313 W   |
    | Inference Energy Consumption | 0.067374 mWh |
    +------------------------------+--------------+[0m
    I0329 14:00:07.487649 139783521261376 runtime_analysis.py:521] 
    Results vgg7-onnx:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.93087    |
    |      Average Precision       |   0.93188    |
    |        Average Recall        |   0.93191    |
    |       Average F1 Score       |   0.93172    |
    |         Average Loss         |   0.22324    |
    |       Average Latency        |  4.9185 ms   |
    |   Average GPU Power Usage    |   49.313 W   |
    | Inference Energy Consumption | 0.067374 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-29/onnx/version_3/model.json[0m
    I0329 14:00:07.489081 139783521261376 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-29/onnx/version_3/model.json
    [32mINFO    [0m [34mSaved mase graph to /root/mase/mase_output/vgg7_cls_cifar10_2024-03-29/software/transform/transformed_ckpt[0m
    I0329 14:06:04.342311 139783521261376 save_and_load.py:147] Saved mase graph to /root/mase/mase_output/vgg7_cls_cifar10_2024-03-29/software/transform/transformed_ckpt
    [32mINFO    [0m [34mTransformation is completed[0m
    I0329 14:06:04.342653 139783521261376 cli.py:388] Transformation is completed


As we can see, the optimized onnx model still outperforms Pytorch on the VGG model due to it's runtime optimizations. The static performs the best, then the automatic mixed precision which outperforms the dynamic quantization due to its requirement of calculating activations on-the-fly. 
