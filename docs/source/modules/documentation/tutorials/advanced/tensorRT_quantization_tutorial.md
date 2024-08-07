# Advanced: TensorRT Quantization Tutorial

This notebook is designed to show the features of the TensorRT passes integrated into MASE as part of the MASERT framework. The following demonstrations were run on a NVIDIA RTX A2000 GPU with a Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz CPU.

## Section 1. INT8 Quantization
Firstly, we will show you how to do a int8 quantization of a simple model, `jsc-toy`, and compare the quantized model to the original model using the `Machop API`. The quantization process is split into the following stages, each using their own individual pass, and are explained in depth at each subsection:

1. [Fake quantization](#section-11-fake-quantization): `tensorrt_fake_quantize_transform_pass`
2. [Calibration](#section-12-calibration): `tensorrt_calibrate_transform_pass`
3. [Quantized Aware Training](#section-13-quantized-aware-training-qat): `tensorrt_fine_tune_transform_pass`
4. [Quantization](#section-14-tensorrt-quantization): `tensorrt_engine_interface_pass`
5. [Analysis](#section-15-performance-analysis): `tensorrt_analysis_pass`

We start by loading in the required libraries and passes required for the notebook as well as ensuring the correct path is set for machop to be used.


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
    tensorrt_calibrate_transform_pass,
    tensorrt_fake_quantize_transform_pass,
    tensorrt_fine_tune_transform_pass,
    tensorrt_engine_interface_pass,
    runtime_analysis_pass,
    )

set_logging_verbosity("info")
```

    [2024-03-29 12:52:18,275] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)


    [32mINFO    [0m [34mSet logging level to info[0m
    WARNING: Logging before flag parsing goes to stderr.
    I0329 12:52:20.742465 139924298352448 logger.py:44] Set logging level to info


Next, we load in the toml file used for quantization. To view the configuration, click [here](../../../machop/configs/tensorrt/jsc_toy_INT8_quantization_by_type.toml).


```python
# Path to your TOML file
JSC_TOML_PATH = '../../../machop/configs/tensorrt/jsc_toy_INT8_quantization_by_type.toml'

# Reading TOML file and converting it into a Python dictionary
with open(JSC_TOML_PATH, 'r') as toml_file:
    pass_args = toml.load(toml_file)

# Extract the 'passes.tensorrt' section and its children
tensorrt_config = pass_args.get('passes', {}).get('tensorrt', {})
# Extract the 'passes.runtime_analysis' section and its children
runtime_analysis_config = pass_args.get('passes', {}).get('runtime_analysis', {})
```

We then create a `MaseGraph` by loading in a model and training it using the toml configuration model arguments.


```python
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
configs = [tensorrt_config, runtime_analysis_config]
for config in configs:
    config['task'] = pass_args['task']
    config['dataset'] = pass_args['dataset']
    config['batch_size'] = pass_args['batch_size']
    config['model'] = pass_args['model']
    config['data_module'] = data_module
    config['accelerator'] = 'cuda' if pass_args['accelerator'] == 'gpu' else pass_args['accelerator']
    if config['accelerator'] == 'gpu':
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

model_info = get_model_info(model_name)
# quant_modules.initialize()
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

Next, we train the `jsc-toy` model using the machop `train` action with the config from the toml file.


```python
!ch train --config {JSC_TOML_PATH}
```

Then we load in the checkpoint. You will have to adjust this according to where it has been stored in the mase_output directory.


```python
# Load in the trained checkpoint - change this accordingly
JSC_CHECKPOINT_PATH = "../../../mase_output/jsc-toy_cls_jsc-pre-trained/best.ckpt"

model = load_model(load_name=JSC_CHECKPOINT_PATH, load_type="pl", model=model)

# Initiate metadata
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)
mg, _ = metadata_value_type_cast_transform_pass(mg, pass_args={"fn": to_numpy_if_tensor})

# Before we begin, we will copy the original MaseGraph model to use for comparison during quantization analysis
mg_original = deepcopy_mase_graph(mg)
```

    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from ../../../mase_output/jsc-toy_cls_jsc-pre-trained/best.ckpt[0m
    I0329 12:52:38.088409 139924298352448 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from ../../../mase_output/jsc-toy_cls_jsc-pre-trained/best.ckpt


### Section 1.1 Fake Quantization

Firstly, we fake quantize the module in order to perform calibration and fine tuning before actually quantizing - this is only used if we have int8 calibration as other precisions are not currently supported within [pytorch-quantization](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#) library.

This is acheived through the `tensorrt_fake_quantize_transform_pass` which goes through the model, either by type or by name, replaces each layer appropriately to a fake quantized form if the `quantize` parameter is set in the default config (`passes.tensorrt.default.config`) or on a per name or type basis. 

Currently the quantizable layers are:
- Linear
- Conv1d, Conv2d, Conv3d 
- ConvTranspose1d, ConvTranspose2d, ConvTranspose3d 
- MaxPool1d, MaxPool2d, MaxPool3d
- AvgPool1d, AvgPool2d, AvgPool3d
- LSTM, LSTMCell

To create a custom quantized module, click [here](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#document-tutorials/creating_custom_quantized_modules).



```python
mg, _ = tensorrt_fake_quantize_transform_pass(mg, pass_args=tensorrt_config)
summarize_quantization_analysis_pass(mg_original, mg)
```

    [32mINFO    [0m [34mApplying fake quantization to PyTorch model...[0m
    I0329 12:52:41.734487 139924298352448 utils.py:282] Applying fake quantization to PyTorch model...
    [32mINFO    [0m [34mFake quantization applied to PyTorch model.[0m
    I0329 12:52:41.836391 139924298352448 utils.py:307] Fake quantization applied to PyTorch model.
    [32mINFO    [0m [34mQuantized graph histogram:[0m
    I0329 12:52:41.848948 139924298352448 summary.py:84] Quantized graph histogram:
    [32mINFO    [0m [34m
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         3 |           0 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |[0m
    I0329 12:52:41.851252 139924298352448 summary.py:85] 
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         3 |           0 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |


As you can see we have succesfully fake quantized all linear layers inside `jsc-toy`. This means that we will be able to simulate a quantized model in order to calibrate and fine tune it. This fake quantization was done on typewise i.e. for linear layers only. See [Section 4](#section-4-layer-wise-mixed-precision) for how to apply quantization layerwise - i.e. only first and second layers for example.

### Section 1.2 Calibration

Next, we perform calibration using the `tensorrt_calibrate_transform_pass`. Calibration is achieved by passing data samples to the quantizer and deciding the best amax for activations. 

Calibrators can be added as a search space parameter to examine the best performing calibrator. The calibrators have been included in the toml as follows.
For example: `calibrators = ["percentile", "mse", "entropy"]`

Note: 
- To use `percentile` calibration, a list of percentiles must be given
- To use `max` calibration, the `histogram` weight and input calibrators must be removed and replaced with `max`. This will use global maximum absolute value to calibrate the model.
- If `post_calibration_analysis` is set true the `tensorrt_analysis_pass` will be run for each calibrator tested to evaluate the most suitable calibrator for the model.


```python
mg, _ = tensorrt_calibrate_transform_pass(mg, pass_args=tensorrt_config)
```

    [32mINFO    [0m [34mStarting calibration of the model in PyTorch...[0m
    I0329 12:52:45.730825 139924298352448 calibrate.py:143] Starting calibration of the model in PyTorch...
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0329 12:52:45.735345 139924298352448 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0329 12:52:45.736729 139924298352448 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0329 12:52:45.738237 139924298352448 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0329 12:52:45.739941 139924298352448 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0329 12:52:45.741892 139924298352448 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0329 12:52:45.743747 139924298352448 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0329 12:52:45.988795 139924298352448 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0329 12:52:45.990496 139924298352448 tensor_quantizer.py:174] Disable MaxCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0329 12:52:45.991268 139924298352448 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0329 12:52:45.992826 139924298352448 tensor_quantizer.py:174] Disable MaxCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0329 12:52:45.993570 139924298352448 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0329 12:52:45.994897 139924298352448 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0329 12:52:45.995635 139924298352448 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0329 12:52:45.997100 139924298352448 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0329 12:52:45.997894 139924298352448 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0329 12:52:45.999178 139924298352448 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0329 12:52:45.999888 139924298352448 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0329 12:52:46.001299 139924298352448 tensor_quantizer.py:174] Disable HistogramCalibrator
    W0329 12:52:46.002087 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    W0329 12:52:46.002799 139924298352448 tensor_quantizer.py:239] Call .cuda() if running on GPU after loading calibrated amax.
    [32mINFO    [0m [34mseq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.3010 calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0329 12:52:46.006715 139924298352448 calibrate.py:131] seq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.3010 calibrator=MaxCalibrator scale=1.0 quant)
    W0329 12:52:46.007978 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([8, 1]).
    [32mINFO    [0m [34mseq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.4018, 0.7529](8) calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0329 12:52:46.009553 139924298352448 calibrate.py:131] seq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.4018, 0.7529](8) calibrator=MaxCalibrator scale=1.0 quant)
    W0329 12:52:46.021312 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.4583 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:46.022117 139924298352448 calibrate.py:131] seq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.4583 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:52:46.024444 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5621 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:46.025047 139924298352448 calibrate.py:131] seq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5621 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:52:46.027429 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=1.7310 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:46.028019 139924298352448 calibrate.py:131] seq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=1.7310 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:52:46.030261 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5606 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:46.030860 139924298352448 calibrate.py:131] seq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5606 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.0...[0m
    I0329 12:52:46.033652 139924298352448 calibrate.py:105] Performing post calibration analysis for calibrator percentile_99.0...
    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy[0m
    I0329 12:52:46.035208 139924298352448 runtime_analysis.py:357] Starting transformation analysis on jsc-toy
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.71232    |
    |      Average Precision       |   0.71349    |
    |        Average Recall        |   0.70401    |
    |       Average F1 Score       |   0.70544    |
    |         Average Loss         |   0.84152    |
    |       Average Latency        |  3.0208 ms   |
    |   Average GPU Power Usage    |   22.239 W   |
    | Inference Energy Consumption | 0.018661 mWh |
    +------------------------------+--------------+[0m
    I0329 12:52:49.573626 139924298352448 runtime_analysis.py:521] 
    Results jsc-toy:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.71232    |
    |      Average Precision       |   0.71349    |
    |        Average Recall        |   0.70401    |
    |       Average F1 Score       |   0.70544    |
    |         Average Loss         |   0.84152    |
    |       Average Latency        |  3.0208 ms   |
    |   Average GPU Power Usage    |   22.239 W   |
    | Inference Energy Consumption | 0.018661 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_1/model.json[0m
    I0329 12:52:49.576186 139924298352448 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_1/model.json
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0329 12:52:49.577105 139924298352448 calibrate.py:118] Post calibration analysis complete.
    W0329 12:52:49.578157 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.3010 calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0329 12:52:49.578797 139924298352448 calibrate.py:131] seq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.3010 calibrator=MaxCalibrator scale=1.0 quant)
    W0329 12:52:49.579839 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([8, 1]).
    [32mINFO    [0m [34mseq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.4018, 0.7529](8) calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0329 12:52:49.580890 139924298352448 calibrate.py:131] seq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.4018, 0.7529](8) calibrator=MaxCalibrator scale=1.0 quant)
    W0329 12:52:49.583162 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=4.0614 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:49.584400 139924298352448 calibrate.py:131] seq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=4.0614 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:52:49.585959 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5621 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:49.586792 139924298352448 calibrate.py:131] seq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5621 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:52:49.588718 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.6858 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:49.589654 139924298352448 calibrate.py:131] seq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.6858 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:52:49.591186 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5606 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:49.592068 139924298352448 calibrate.py:131] seq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5606 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.9...[0m
    I0329 12:52:49.594088 139924298352448 calibrate.py:105] Performing post calibration analysis for calibrator percentile_99.9...
    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy[0m
    I0329 12:52:49.595173 139924298352448 runtime_analysis.py:357] Starting transformation analysis on jsc-toy
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.71678    |
    |      Average Precision       |   0.71959    |
    |        Average Recall        |   0.71039    |
    |       Average F1 Score       |   0.71209    |
    |         Average Loss         |   0.81512    |
    |       Average Latency        |  3.0252 ms   |
    |   Average GPU Power Usage    |   22.21 W    |
    | Inference Energy Consumption | 0.018664 mWh |
    +------------------------------+--------------+[0m
    I0329 12:52:53.233150 139924298352448 runtime_analysis.py:521] 
    Results jsc-toy:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.71678    |
    |      Average Precision       |   0.71959    |
    |        Average Recall        |   0.71039    |
    |       Average F1 Score       |   0.71209    |
    |         Average Loss         |   0.81512    |
    |       Average Latency        |  3.0252 ms   |
    |   Average GPU Power Usage    |   22.21 W    |
    | Inference Energy Consumption | 0.018664 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_2/model.json[0m
    I0329 12:52:53.235970 139924298352448 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_2/model.json
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0329 12:52:53.237108 139924298352448 calibrate.py:118] Post calibration analysis complete.
    W0329 12:52:53.238376 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.3010 calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0329 12:52:53.240151 139924298352448 calibrate.py:131] seq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.3010 calibrator=MaxCalibrator scale=1.0 quant)
    W0329 12:52:53.241358 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([8, 1]).
    [32mINFO    [0m [34mseq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.4018, 0.7529](8) calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0329 12:52:53.242765 139924298352448 calibrate.py:131] seq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.4018, 0.7529](8) calibrator=MaxCalibrator scale=1.0 quant)
    W0329 12:52:53.244726 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=4.9840 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:53.245705 139924298352448 calibrate.py:131] seq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=4.9840 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:52:53.246822 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5621 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:53.247384 139924298352448 calibrate.py:131] seq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5621 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:52:53.248608 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=3.0583 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:53.249203 139924298352448 calibrate.py:131] seq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=3.0583 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:52:53.250317 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5606 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:53.250878 139924298352448 calibrate.py:131] seq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5606 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.99...[0m
    I0329 12:52:53.252007 139924298352448 calibrate.py:105] Performing post calibration analysis for calibrator percentile_99.99...
    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy[0m
    I0329 12:52:53.252766 139924298352448 runtime_analysis.py:357] Starting transformation analysis on jsc-toy
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.71766    |
    |      Average Precision       |   0.72036    |
    |        Average Recall        |   0.71136    |
    |       Average F1 Score       |   0.71308    |
    |         Average Loss         |   0.81424    |
    |       Average Latency        |  2.8614 ms   |
    |   Average GPU Power Usage    |   22.252 W   |
    | Inference Energy Consumption | 0.017687 mWh |
    +------------------------------+--------------+[0m
    I0329 12:52:56.441818 139924298352448 runtime_analysis.py:521] 
    Results jsc-toy:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.71766    |
    |      Average Precision       |   0.72036    |
    |        Average Recall        |   0.71136    |
    |       Average F1 Score       |   0.71308    |
    |         Average Loss         |   0.81424    |
    |       Average Latency        |  2.8614 ms   |
    |   Average GPU Power Usage    |   22.252 W   |
    | Inference Energy Consumption | 0.017687 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_3/model.json[0m
    I0329 12:52:56.444185 139924298352448 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_3/model.json
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0329 12:52:56.445506 139924298352448 calibrate.py:118] Post calibration analysis complete.
    W0329 12:52:56.446318 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.3010 calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0329 12:52:56.447259 139924298352448 calibrate.py:131] seq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.3010 calibrator=MaxCalibrator scale=1.0 quant)
    W0329 12:52:56.448057 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([8, 1]).
    [32mINFO    [0m [34mseq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.4018, 0.7529](8) calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0329 12:52:56.448869 139924298352448 calibrate.py:131] seq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.4018, 0.7529](8) calibrator=MaxCalibrator scale=1.0 quant)
    W0329 12:52:57.950802 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=4.9235 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:57.952084 139924298352448 calibrate.py:131] seq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=4.9235 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:52:59.105022 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5601 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:52:59.106170 139924298352448 calibrate.py:131] seq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5601 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:53:00.794175 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=3.0265 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:53:00.795263 139924298352448 calibrate.py:131] seq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=3.0265 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:53:01.958939 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5588 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:53:01.959860 139924298352448 calibrate.py:131] seq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5588 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator mse...[0m
    I0329 12:53:01.961093 139924298352448 calibrate.py:105] Performing post calibration analysis for calibrator mse...
    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy[0m
    I0329 12:53:01.961857 139924298352448 runtime_analysis.py:357] Starting transformation analysis on jsc-toy
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.71739    |
    |      Average Precision       |   0.72013    |
    |        Average Recall        |   0.71102    |
    |       Average F1 Score       |   0.71269    |
    |         Average Loss         |   0.81419    |
    |       Average Latency        |  2.9989 ms   |
    |   Average GPU Power Usage    |   22.426 W   |
    | Inference Energy Consumption | 0.018681 mWh |
    +------------------------------+--------------+[0m
    I0329 12:53:05.428555 139924298352448 runtime_analysis.py:521] 
    Results jsc-toy:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.71739    |
    |      Average Precision       |   0.72013    |
    |        Average Recall        |   0.71102    |
    |       Average F1 Score       |   0.71269    |
    |         Average Loss         |   0.81419    |
    |       Average Latency        |  2.9989 ms   |
    |   Average GPU Power Usage    |   22.426 W   |
    | Inference Energy Consumption | 0.018681 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_4/model.json[0m
    I0329 12:53:05.430769 139924298352448 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_4/model.json
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0329 12:53:05.431792 139924298352448 calibrate.py:118] Post calibration analysis complete.
    W0329 12:53:05.433431 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.3010 calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0329 12:53:05.434180 139924298352448 calibrate.py:131] seq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.3010 calibrator=MaxCalibrator scale=1.0 quant)
    W0329 12:53:05.435153 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([8, 1]).
    [32mINFO    [0m [34mseq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.4018, 0.7529](8) calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0329 12:53:05.436254 139924298352448 calibrate.py:131] seq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.4018, 0.7529](8) calibrator=MaxCalibrator scale=1.0 quant)
    W0329 12:53:09.528398 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=4.7816 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:53:09.529677 139924298352448 calibrate.py:131] seq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=4.7816 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:53:11.763471 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5624 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:53:11.764446 139924298352448 calibrate.py:131] seq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5624 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:53:17.317717 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=3.0593 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:53:17.318790 139924298352448 calibrate.py:131] seq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=3.0593 calibrator=HistogramCalibrator scale=1.0 quant)
    W0329 12:53:19.543643 139924298352448 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5609 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0329 12:53:19.544808 139924298352448 calibrate.py:131] seq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5609 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator entropy...[0m
    I0329 12:53:19.546923 139924298352448 calibrate.py:105] Performing post calibration analysis for calibrator entropy...
    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy[0m
    I0329 12:53:19.548315 139924298352448 runtime_analysis.py:357] Starting transformation analysis on jsc-toy
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.71737    |
    |      Average Precision       |   0.72008    |
    |        Average Recall        |   0.71095    |
    |       Average F1 Score       |   0.71263    |
    |         Average Loss         |   0.81421    |
    |       Average Latency        |  2.9006 ms   |
    |   Average GPU Power Usage    |   22.525 W   |
    | Inference Energy Consumption | 0.018149 mWh |
    +------------------------------+--------------+[0m
    I0329 12:53:22.697756 139924298352448 runtime_analysis.py:521] 
    Results jsc-toy:
    +------------------------------+--------------+
    |      Metric (Per Batch)      |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.71737    |
    |      Average Precision       |   0.72008    |
    |        Average Recall        |   0.71095    |
    |       Average F1 Score       |   0.71263    |
    |         Average Loss         |   0.81421    |
    |       Average Latency        |  2.9006 ms   |
    |   Average GPU Power Usage    |   22.525 W   |
    | Inference Energy Consumption | 0.018149 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_5/model.json[0m
    I0329 12:53:22.699659 139924298352448 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_5/model.json
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0329 12:53:22.700434 139924298352448 calibrate.py:118] Post calibration analysis complete.
    [32mINFO    [0m [34mSucceeded in calibrating the model in PyTorch![0m
    I0329 12:53:22.701544 139924298352448 calibrate.py:213] Succeeded in calibrating the model in PyTorch!


From the results, the 99% `percentile` clips too many values during the amax calibration, compromising the loss. However 99.99% demonstrates higher validation accuracy alongside `mse` and `entropy` for `jsc-toy`. For such a small model, the methods are not highly distinguished, however for larger models this calibration process will be important for ensuring the quantized model still performs well. 

### Section 1.3 Quantized Aware Training (QAT)

The `tensorrt_fine_tune_transform_pass` is used to fine tune the quantized model. By default, when running the `tensorrt_engine_interface_pass` the fake quantized model will go through fine tuning however you stop this by setting the `fine_tune` in `passes.tensorrt.fine_tune` to false.

For QAT it is typical to employ 10% of the original training epochs, starting at 1% of the initial training learning rate, and a cosine annealing learning rate schedule that follows the decreasing half of a cosine period, down to 1% of the initial fine tuning learning rate (0.01% of the initial training learning rate). However this default can be overidden by setting the `epochs`, `initial_learning_rate` and `final_learning_rate` in `passes.tensorrt.fine_tune`.

The fine tuned checkpoints are stored in the ckpts/fine_tuning folder:

```
mase_output
└── tensorrt
    └── quantization
        └──model_task_dataset_date
            ├── cache
            ├── ckpts
            │   └── fine_tuning
            ├── json
            ├── onnx
            └── trt
```


```python
mg, _ = tensorrt_fine_tune_transform_pass(mg, pass_args=tensorrt_config)
```

    [32mINFO    [0m [34mStarting Fine Tuning for 2 epochs...[0m
    I0329 12:53:56.899361 139924298352448 fine_tune.py:142] Starting Fine Tuning for 2 epochs...
    I0329 12:53:57.030875 139924298352448 rank_zero.py:64] GPU available: True (cuda), used: True
    I0329 12:53:57.054269 139924298352448 rank_zero.py:64] TPU available: False, using: 0 TPU cores
    I0329 12:53:57.055013 139924298352448 rank_zero.py:64] IPU available: False, using: 0 IPUs
    I0329 12:53:57.055576 139924298352448 rank_zero.py:64] HPU available: False, using: 0 HPUs
    I0329 12:53:57.062752 139924298352448 rank_zero.py:64] You are using a CUDA device ('NVIDIA RTX A2000 12GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision


    I0329 12:53:59.800536 139924298352448 cuda.py:61] LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    I0329 12:53:59.814722 139924298352448 model_summary.py:94] 
      | Name      | Type               | Params
    -------------------------------------------------
    0 | model     | GraphModule        | 327   
    1 | loss_fn   | CrossEntropyLoss   | 0     
    2 | acc_train | MulticlassAccuracy | 0     
    3 | loss_val  | MeanMetric         | 0     
    4 | loss_test | MeanMetric         | 0     
    -------------------------------------------------
    327       Trainable params
    0         Non-trainable params
    327       Total params
    0.001     Total estimated model params size (MB)



    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:441: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=27` in the `DataLoader` to improve performance.
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=27` in the `DataLoader` to improve performance.



    Training: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]


    I0329 13:00:33.763407 139924298352448 rank_zero.py:64] `Trainer.fit` stopped: `max_epochs=2` reached.
    [32mINFO    [0m [34mFine Tuning Complete[0m
    I0329 13:00:33.770675 139924298352448 fine_tune.py:161] Fine Tuning Complete


### Section 1.4 TensorRT Quantization

After QAT, we are now ready to convert the model to a tensorRT engine so that it can be run with the superior inference speeds. To do so, we use the `tensorrt_engine_interface_pass` which converts the `MaseGraph`'s model from a Pytorch one to an ONNX format as an intermediate stage of the conversion.

During the conversion process, the `.onnx` and `.trt` files are stored to their respective folders shown in [Section 1.3](#section-13-quantized-aware-training-qat).

This interface pass returns a dictionary containing the `onnx_path` and `trt_engine_path`.


```python
mg, meta = tensorrt_engine_interface_pass(mg, pass_args=tensorrt_config)
```

    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0329 13:02:33.433028 139924298352448 quantize.py:209] Converting PyTorch model to ONNX...
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:363: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax < 0:
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:366: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      max_bound = torch.tensor((2.0**(num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:376: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:382: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax <= epsilon:
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/2024-03-29/version_1/model.onnx[0m
    I0329 13:02:33.677475 139924298352448 quantize.py:239] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/2024-03-29/version_1/model.onnx
    [32mINFO    [0m [34mConverting PyTorch model to TensorRT...[0m
    I0329 13:02:33.678984 139924298352448 quantize.py:102] Converting PyTorch model to TensorRT...
    [32mINFO    [0m [34mTensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/2024-03-29/version_2/model.trt[0m
    I0329 13:03:29.229270 139924298352448 quantize.py:202] TensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/2024-03-29/version_2/model.trt
    [32mINFO    [0m [34mTensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/2024-03-29/version_3/model.json[0m
    I0329 13:03:29.479406 139924298352448 quantize.py:259] TensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/2024-03-29/version_3/model.json


### Section 1.5 Performance Analysis

To showcase the improved inference speeds and to evaluate accuracy and other performance metrics, the `tensorrt_analysis_pass` can be used.

The tensorRT engine path obtained the previous interface pass is now inputted into the the analysis pass. The same pass can take a MaseGraph as an input, as well as an ONNX graph. For this comparison, we will first run the anaylsis pass on the original unquantized model and then on the int8 quantized model.


```python
_, _ = runtime_analysis_pass(mg_original, pass_args=runtime_analysis_config)
_, _ = runtime_analysis_pass(meta['trt_engine_path'], pass_args=runtime_analysis_config)
```

    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy[0m
    I0329 13:03:29.966202 139924298352448 runtime_analysis.py:357] Starting transformation analysis on jsc-toy
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+---------------+
    |      Metric (Per Batch)      |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.71971    |
    |      Average Precision       |    0.71884    |
    |        Average Recall        |    0.71127    |
    |       Average F1 Score       |    0.71274    |
    |         Average Loss         |    0.8116     |
    |       Average Latency        |  0.87057 ms   |
    |   Average GPU Power Usage    |   23.792 W    |
    | Inference Energy Consumption | 0.0057535 mWh |
    +------------------------------+---------------+[0m
    I0329 13:03:32.504800 139924298352448 runtime_analysis.py:521] 
    Results jsc-toy:
    +------------------------------+---------------+
    |      Metric (Per Batch)      |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.71971    |
    |      Average Precision       |    0.71884    |
    |        Average Recall        |    0.71127    |
    |       Average F1 Score       |    0.71274    |
    |         Average Loss         |    0.8116     |
    |       Average Latency        |  0.87057 ms   |
    |   Average GPU Power Usage    |   23.792 W    |
    | Inference Energy Consumption | 0.0057535 mWh |
    +------------------------------+---------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_6/model.json[0m
    I0329 13:03:32.507777 139924298352448 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/mase_graph/version_6/model.json
    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy-trt_quantized[0m
    I0329 13:03:32.523447 139924298352448 runtime_analysis.py:357] Starting transformation analysis on jsc-toy-trt_quantized
    [32mINFO    [0m [34m
    Results jsc-toy-trt_quantized:
    +------------------------------+----------------+
    |      Metric (Per Batch)      |     Value      |
    +------------------------------+----------------+
    |    Average Test Accuracy     |    0.73069     |
    |      Average Precision       |    0.74101     |
    |        Average Recall        |    0.72967     |
    |       Average F1 Score       |    0.73247     |
    |         Average Loss         |    0.76993     |
    |       Average Latency        |   0.13363 ms   |
    |   Average GPU Power Usage    |    23.043 W    |
    | Inference Energy Consumption | 0.00085532 mWh |
    +------------------------------+----------------+[0m
    I0329 13:03:34.503784 139924298352448 runtime_analysis.py:521] 
    Results jsc-toy-trt_quantized:
    +------------------------------+----------------+
    |      Metric (Per Batch)      |     Value      |
    +------------------------------+----------------+
    |    Average Test Accuracy     |    0.73069     |
    |      Average Precision       |    0.74101     |
    |        Average Recall        |    0.72967     |
    |       Average F1 Score       |    0.73247     |
    |         Average Loss         |    0.76993     |
    |       Average Latency        |   0.13363 ms   |
    |   Average GPU Power Usage    |    23.043 W    |
    | Inference Energy Consumption | 0.00085532 mWh |
    +------------------------------+----------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/tensorrt/version_0/model.json[0m
    I0329 13:03:34.506492 139924298352448 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-29/tensorrt/version_0/model.json


As shown above, the latency has decreased around 6x with the `jsc-toy` model without compromising accuracy due to the well calibrated amax and quantization-aware fine tuning and additional runtime optimizations from TensorRT. The inference energy consumption has thus also dropped tremendously and this is an excellent demonstration for the need to quantize in industry especially for LLMs in order to reduce energy usage. 

## Section 2. FP16 Quantization

We will now load in a new toml configuration that uses fp16 instead of int8, whilst keeping the other settings the exact same for a fair comparison. This time however, we will use chop from the terminal which runs all the passes showcased in [Section 1](#section-1---int8-quantization).

Since float quantization does not require calibration, nor is it supported by `pytorch-quantization`, the model will not undergo fake quantization; for the time being this unfortunately means QAT is unavailable and only undergoes Post Training Quantization (PTQ). 


```python
JSC_FP16_BY_TYPE_TOML = "../../../machop/configs/tensorrt/jsc_toy_FP16_quantization_by_type.toml"
!ch transform --config {JSC_FP16_BY_TYPE_TOML} --load {JSC_CHECKPOINT_PATH} --load-type pl
```

    8808.24s - pydevd: Sending message related to process being replaced timed-out after 5 seconds
    [2024-03-28 09:37:03,989] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0328 09:37:06.938809 140201001654080 seed.py:54] Seed set to 0
    +-------------------------+------------------------+--------------------------+--------------------------+--------------------------+
    | Name                    |        Default         |       Config. File       |     Manual Override      |        Effective         |
    +-------------------------+------------------------+--------------------------+--------------------------+--------------------------+
    | task                    |     [38;5;8mclassification[0m     |           cls            |                          |           cls            |
    | load_name               |          [38;5;8mNone[0m          | [38;5;8m../mase_output/jsc-toy_c[0m | /root/mase/mase_output/j | /root/mase/mase_output/j |
    |                         |                        | [38;5;8mls_jsc/software/training[0m |   sc-toy_cls_jsc-pre-    |   sc-toy_cls_jsc-pre-    |
    |                         |                        |     [38;5;8m_ckpts/best.ckpt[0m     |    trained/best.ckpt     |    trained/best.ckpt     |
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
    | model                   |          [38;5;8mNone[0m          |         jsc-toy          |                          |         jsc-toy          |
    | dataset                 |          [38;5;8mNone[0m          |           jsc            |                          |           jsc            |
    | t_max                   |           20           |                          |                          |            20            |
    | eta_min                 |         1e-06          |                          |                          |          1e-06           |
    +-------------------------+------------------------+--------------------------+--------------------------+--------------------------+
    [32mINFO    [0m [34mInitialising model 'jsc-toy'...[0m
    I0328 09:37:06.948820 140201001654080 cli.py:841] Initialising model 'jsc-toy'...
    [32mINFO    [0m [34mInitialising dataset 'jsc'...[0m
    I0328 09:37:06.950793 140201001654080 cli.py:869] Initialising dataset 'jsc'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-28[0m
    I0328 09:37:06.951038 140201001654080 cli.py:905] Project will be created at /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-28
    [32mINFO    [0m [34mTransforming model 'jsc-toy'...[0m
    I0328 09:37:07.078488 140201001654080 cli.py:365] Transforming model 'jsc-toy'...
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/jsc-toy_cls_jsc-pre-trained/best.ckpt[0m
    I0328 09:37:09.420990 140201001654080 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/jsc-toy_cls_jsc-pre-trained/best.ckpt
    [32mINFO    [0m [34mApplying fake quantization to PyTorch model...[0m
    I0328 09:37:11.531935 140201001654080 utils.py:240] Applying fake quantization to PyTorch model...
    [33mWARNING [0m [34mint8 precision not found in config. Skipping fake quantization.[0m
    W0328 09:37:11.532341 140201001654080 utils.py:243] int8 precision not found in config. Skipping fake quantization.
    [32mINFO    [0m [34mQuantized graph histogram:[0m
    I0328 09:37:11.550864 140201001654080 summary.py:84] Quantized graph histogram:
    [32mINFO    [0m [34m
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         0 |           3 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |[0m
    I0328 09:37:11.551648 140201001654080 summary.py:85] 
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         0 |           3 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |
    [33mWARNING [0m [34mint8 precision not found in config. Skipping calibration.[0m
    W0328 09:37:11.552517 140201001654080 calibrate.py:137] int8 precision not found in config. Skipping calibration.
    [33mWARNING [0m [34mint8 precision not found in config. Skipping QAT fine tuning.[0m
    W0328 09:37:11.553805 140201001654080 fine_tune.py:92] int8 precision not found in config. Skipping QAT fine tuning.
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0328 09:37:11.556088 140201001654080 quantize.py:171] Converting PyTorch model to ONNX...
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-28/2024-03-28/version_1/model.onnx[0m
    I0328 09:37:13.650603 140201001654080 quantize.py:194] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-28/2024-03-28/version_1/model.onnx
    [32mINFO    [0m [34mConverting PyTorch model to TensorRT...[0m
    I0328 09:37:13.651079 140201001654080 quantize.py:97] Converting PyTorch model to TensorRT...
    [32mINFO    [0m [34mTensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-28/2024-03-28/version_2/model.trt[0m
    I0328 09:37:30.438357 140201001654080 quantize.py:166] TensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-28/2024-03-28/version_2/model.trt
    [32mINFO    [0m [34mTensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-28/2024-03-28/version_3/model.json[0m
    I0328 09:37:30.664676 140201001654080 quantize.py:210] TensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-28/2024-03-28/version_3/model.json
    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy[0m
    I0328 09:37:30.666585 140201001654080 runtime_analysis.py:309] Starting transformation analysis on jsc-toy
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+---------------+
    |      Metric (Per Batch)      |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.71971    |
    |      Average Precision       |    0.71884    |
    |        Average Recall        |    0.71127    |
    |       Average F1 Score       |    0.71274    |
    |         Average Loss         |    0.8116     |
    |       Average Latency        |  0.80336 ms   |
    |   Average GPU Power Usage    |   22.024 W    |
    | Inference Energy Consumption | 0.0049148 mWh |
    +------------------------------+---------------+[0m
    I0328 09:37:36.951404 140201001654080 runtime_analysis.py:437] 
    Results jsc-toy:
    +------------------------------+---------------+
    |      Metric (Per Batch)      |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.71971    |
    |      Average Precision       |    0.71884    |
    |        Average Recall        |    0.71127    |
    |       Average F1 Score       |    0.71274    |
    |         Average Loss         |    0.8116     |
    |       Average Latency        |  0.80336 ms   |
    |   Average GPU Power Usage    |   22.024 W    |
    | Inference Energy Consumption | 0.0049148 mWh |
    +------------------------------+---------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-28/mase_graph/version_43/model.json[0m
    I0328 09:37:36.952783 140201001654080 runtime_analysis.py:123] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-28/mase_graph/version_43/model.json
    [32mINFO    [0m [34m
    TensorRT Engine Input/Output Information:
    Index | Type    | DataType | Static Shape         | Dynamic Shape        | Name
    ------|---------|----------|----------------------|----------------------|-----------------------
    0     | Input   | FLOAT    | (64, 16)               | (64, 16)               | input
    1     | Output  | FLOAT    | (64, 5)                | (64, 5)                | 37[0m
    I0328 09:37:36.960667 140201001654080 runtime_analysis.py:167] 
    TensorRT Engine Input/Output Information:
    Index | Type    | DataType | Static Shape         | Dynamic Shape        | Name
    ------|---------|----------|----------------------|----------------------|-----------------------
    0     | Input   | FLOAT    | (64, 16)               | (64, 16)               | input
    1     | Output  | FLOAT    | (64, 5)                | (64, 5)                | 37
    [32mINFO    [0m [34mStarting transformation analysis on jsc-toy-trt_quantized[0m
    I0328 09:37:36.960840 140201001654080 runtime_analysis.py:309] Starting transformation analysis on jsc-toy-trt_quantized
    [32mINFO    [0m [34m
    Results jsc-toy-trt_quantized:
    +------------------------------+----------------+
    |      Metric (Per Batch)      |     Value      |
    +------------------------------+----------------+
    |    Average Test Accuracy     |    0.73639     |
    |      Average Precision       |    0.74849     |
    |        Average Recall        |    0.73504     |
    |       Average F1 Score       |    0.73822     |
    |         Average Loss         |    0.74597     |
    |       Average Latency        |   0.09133 ms   |
    |   Average GPU Power Usage    |    21.706 W    |
    | Inference Energy Consumption | 0.00055067 mWh |
    +------------------------------+----------------+[0m
    I0328 09:37:43.052305 140201001654080 runtime_analysis.py:437] 
    Results jsc-toy-trt_quantized:
    +------------------------------+----------------+
    |      Metric (Per Batch)      |     Value      |
    +------------------------------+----------------+
    |    Average Test Accuracy     |    0.73639     |
    |      Average Precision       |    0.74849     |
    |        Average Recall        |    0.73504     |
    |       Average F1 Score       |    0.73822     |
    |         Average Loss         |    0.74597     |
    |       Average Latency        |   0.09133 ms   |
    |   Average GPU Power Usage    |    21.706 W    |
    | Inference Energy Consumption | 0.00055067 mWh |
    +------------------------------+----------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-28/tensorrt/version_1/model.json[0m
    I0328 09:37:43.054715 140201001654080 runtime_analysis.py:123] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/jsc-toy_cls_jsc_2024-03-28/tensorrt/version_1/model.json
    [32mINFO    [0m [34mSaved mase graph to /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-28/software/transform/transformed_ckpt[0m
    I0328 09:37:43.132117 140201001654080 save_and_load.py:147] Saved mase graph to /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-28/software/transform/transformed_ckpt
    [32mINFO    [0m [34mTransformation is completed[0m
    I0328 09:37:43.132461 140201001654080 cli.py:383] Transformation is completed


As you can see, `fp16` acheives a slighty higher test accuracy but a slightly lower latency (~30%) from that of int8 quantization; it is still ~2.5x faster than the unquantized model. Now lets apply quantization to a more complicated model.

## Section 3. Type-wise Mixed Precision on Larger Model
We will now quantize `vgg7` which includes both convolutional and linear layers, however for this demonstration we want to quantize all layer types except the linear layers.

In this case, we set:

- The `by` parameter to `type`
- The `quantize` parameter to true for `passes.tensorrt.conv2d.config` and `precision` parameter to 'int8'.
- The `input` and `weight` quantize axis for the conv2d layers.
- The default `passes.tensorrt.default.config` precision to true. 

During the TensorRT quantization, the model's conv2d layers will be converted to an int8 fake quantized form, whilst the linear layers are kept to their default 'fp16'. Calibration of the conv2d layers and then fine tuning will be undergone before quantization and inference.

You may either download a pretrained model [here](https://imperiallondon-my.sharepoint.com/:f:/g/personal/zz7522_ic_ac_uk/Emh3VT7Q_qRFmnp8kDrcgDoBwGUuzLwwKNtX8ZAt368jJQ?e=gsKONa), otherwise train it yourself as shown below. 


```python
VGG_TYPEWISE_TOML = "../../../machop/configs/tensorrt/vgg7_typewise_mixed_precision.toml"

!ch train --config {VGG_TYPEWISE_TOML}
```

We will now load the checkpoint in, quantize the model and compare it to the unquantized version as we did in [Section 1.5](#section-15-performance-analysis)


```python
# Change this checkpoint path accordingly
VGG_CHECKPOINT_PATH = "../../../mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt"
```


```python
!ch transform --config {VGG_TYPEWISE_TOML} --load {VGG_CHECKPOINT_PATH} --load-type pl
```

    [2024-03-28 23:00:09,016] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0328 23:00:12.031970 139939454809920 seed.py:54] Seed set to 0
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
    I0328 23:00:12.042508 139939454809920 cli.py:846] Initialising model 'vgg7'...
    [32mINFO    [0m [34mInitialising dataset 'cifar10'...[0m
    I0328 23:00:12.149944 139939454809920 cli.py:874] Initialising dataset 'cifar10'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-28[0m
    I0328 23:00:12.150315 139939454809920 cli.py:910] Project will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-28
    [32mINFO    [0m [34mTransforming model 'vgg7'...[0m
    I0328 23:00:12.277644 139939454809920 cli.py:370] Transforming model 'vgg7'...
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0328 23:00:18.166216 139939454809920 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0328 23:00:18.288503 139939454809920 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt
    [32mINFO    [0m [34mApplying fake quantization to PyTorch model...[0m
    I0328 23:00:36.982592 139939454809920 utils.py:282] Applying fake quantization to PyTorch model...
    [32mINFO    [0m [34mFake quantization applied to PyTorch model.[0m
    I0328 23:00:37.249866 139939454809920 utils.py:307] Fake quantization applied to PyTorch model.
    [32mINFO    [0m [34mQuantized graph histogram:[0m
    I0328 23:00:37.269716 139939454809920 summary.py:84] Quantized graph histogram:
    [32mINFO    [0m [34m
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm2d     | batch_norm2d |       6 |         0 |           6 |
    | Conv2d          | conv2d       |       6 |         6 |           0 |
    | Linear          | linear       |       3 |         0 |           3 |
    | MaxPool2d       | max_pool2d   |       3 |         0 |           3 |
    | ReLU            | relu         |       8 |         0 |           8 |
    | output          | output       |       1 |         0 |           1 |
    | view            | view         |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |[0m
    I0328 23:00:37.270473 139939454809920 summary.py:85] 
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm2d     | batch_norm2d |       6 |         0 |           6 |
    | Conv2d          | conv2d       |       6 |         6 |           0 |
    | Linear          | linear       |       3 |         0 |           3 |
    | MaxPool2d       | max_pool2d   |       3 |         0 |           3 |
    | ReLU            | relu         |       8 |         0 |           8 |
    | output          | output       |       1 |         0 |           1 |
    | view            | view         |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |
    [32mINFO    [0m [34mStarting calibration of the model in PyTorch...[0m
    I0328 23:00:37.271291 139939454809920 calibrate.py:143] Starting calibration of the model in PyTorch...
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.301312 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.301464 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.301600 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.301703 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.301807 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.301904 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.302003 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.302098 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.302198 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.302290 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.302384 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:00:37.302474 139939454809920 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.843650 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.844024 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.844100 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.844196 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.844268 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.844380 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.844432 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.844521 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.844587 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.844675 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.844727 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.844815 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.844870 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.844956 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.844999 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.845094 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.845181 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.845281 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.845329 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.845412 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.845459 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.845543 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:00:43.845585 139939454809920 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:00:43.845667 139939454809920 tensor_quantizer.py:174] Disable HistogramCalibrator
    W0328 23:00:43.853768 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    W0328 23:00:43.853853 139939454809920 tensor_quantizer.py:239] Call .cuda() if running on GPU after loading calibrated amax.
    [32mINFO    [0m [34mfeature_layers.0._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=2.6051 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.853991 139939454809920 calibrate.py:131] feature_layers.0._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=2.6051 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.854280 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.0._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2797 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.854381 139939454809920 calibrate.py:131] feature_layers.0._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2797 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.854688 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.3027 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.854788 139939454809920 calibrate.py:131] feature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.3027 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.855069 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2366 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.855164 139939454809920 calibrate.py:131] feature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2366 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.855502 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=1.8357 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.855595 139939454809920 calibrate.py:131] feature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=1.8357 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.855849 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2296 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.855940 139939454809920 calibrate.py:131] feature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2296 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.856252 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.4749 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.856347 139939454809920 calibrate.py:131] feature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.4749 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.856633 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2080 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.856730 139939454809920 calibrate.py:131] feature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2080 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.857007 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.9279 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.857099 139939454809920 calibrate.py:131] feature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.9279 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.857369 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2013 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.857461 139939454809920 calibrate.py:131] feature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2013 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.857726 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.6148 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.857817 139939454809920 calibrate.py:131] feature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.6148 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:43.858080 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.1879 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:43.858170 139939454809920 calibrate.py:131] feature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.1879 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.0...[0m
    I0328 23:00:43.858960 139939454809920 calibrate.py:105] Performing post calibration analysis for calibrator percentile_99.0...
    [32mINFO    [0m [34mStarting transformation analysis on vgg7[0m
    I0328 23:00:43.859171 139939454809920 runtime_analysis.py:357] Starting transformation analysis on vgg7
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.91305   |
    |      Average Precision       |   0.91207   |
    |        Average Recall        |   0.91246   |
    |       Average F1 Score       |   0.9122    |
    |         Average Loss         |   0.26363   |
    |       Average Latency        |  15.113 ms  |
    |   Average GPU Power Usage    |  59.019 W   |
    | Inference Energy Consumption | 0.24777 mWh |
    +------------------------------+-------------+[0m
    I0328 23:00:55.766893 139939454809920 runtime_analysis.py:521] 
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.91305   |
    |      Average Precision       |   0.91207   |
    |        Average Recall        |   0.91246   |
    |       Average F1 Score       |   0.9122    |
    |         Average Loss         |   0.26363   |
    |       Average Latency        |  15.113 ms  |
    |   Average GPU Power Usage    |  59.019 W   |
    | Inference Energy Consumption | 0.24777 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_44/model.json[0m
    I0328 23:00:55.769451 139939454809920 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_44/model.json
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0328 23:00:55.769782 139939454809920 calibrate.py:118] Post calibration analysis complete.
    W0328 23:00:55.770918 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.0._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=2.6381 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.771324 139939454809920 calibrate.py:131] feature_layers.0._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=2.6381 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.772012 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.0._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3434 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.772331 139939454809920 calibrate.py:131] feature_layers.0._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3434 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.773109 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=5.9141 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.773458 139939454809920 calibrate.py:131] feature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=5.9141 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.774124 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3704 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.774437 139939454809920 calibrate.py:131] feature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3704 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.775206 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.2644 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.775516 139939454809920 calibrate.py:131] feature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.2644 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.776169 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3621 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.776488 139939454809920 calibrate.py:131] feature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3621 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.777268 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.4170 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.777583 139939454809920 calibrate.py:131] feature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.4170 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.778250 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2821 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.778557 139939454809920 calibrate.py:131] feature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2821 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.779228 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.9863 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.779539 139939454809920 calibrate.py:131] feature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.9863 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.780190 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2734 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.780498 139939454809920 calibrate.py:131] feature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2734 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.781196 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.7147 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.781511 139939454809920 calibrate.py:131] feature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.7147 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:00:55.782173 139939454809920 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2519 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:00:55.782480 139939454809920 calibrate.py:131] feature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2519 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.9...[0m
    I0328 23:00:55.783563 139939454809920 calibrate.py:105] Performing post calibration analysis for calibrator percentile_99.9...
    [32mINFO    [0m [34mStarting transformation analysis on vgg7[0m
    I0328 23:00:55.783894 139939454809920 runtime_analysis.py:357] Starting transformation analysis on vgg7
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.92028   |
    |      Average Precision       |   0.91911   |
    |        Average Recall        |   0.9195    |
    |       Average F1 Score       |   0.91919   |
    |         Average Loss         |   0.24024   |
    |       Average Latency        |  15.278 ms  |
    |   Average GPU Power Usage    |  59.653 W   |
    | Inference Energy Consumption | 0.25317 mWh |
    +------------------------------+-------------+[0m
    I0328 23:01:07.450706 139939454809920 runtime_analysis.py:521] 
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.92028   |
    |      Average Precision       |   0.91911   |
    |        Average Recall        |   0.9195    |
    |       Average F1 Score       |   0.91919   |
    |         Average Loss         |   0.24024   |
    |       Average Latency        |  15.278 ms  |
    |   Average GPU Power Usage    |  59.653 W   |
    | Inference Energy Consumption | 0.25317 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_45/model.json[0m
    I0328 23:01:07.452143 139939454809920 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_45/model.json
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0328 23:01:07.452330 139939454809920 calibrate.py:118] Post calibration analysis complete.
    [32mINFO    [0m [34mSucceeded in calibrating the model in PyTorch![0m
    I0328 23:01:07.452472 139939454809920 calibrate.py:213] Succeeded in calibrating the model in PyTorch!
    [32mINFO    [0m [34mStarting Fine Tuning for 2 epochs...[0m
    I0328 23:01:07.456390 139939454809920 fine_tune.py:142] Starting Fine Tuning for 2 epochs...
    INFO: GPU available: True (cuda), used: True
    I0328 23:01:07.618809 139939454809920 rank_zero.py:64] GPU available: True (cuda), used: True
    INFO: TPU available: False, using: 0 TPU cores
    I0328 23:01:07.645764 139939454809920 rank_zero.py:64] TPU available: False, using: 0 TPU cores
    INFO: IPU available: False, using: 0 IPUs
    I0328 23:01:07.645846 139939454809920 rank_zero.py:64] IPU available: False, using: 0 IPUs
    INFO: HPU available: False, using: 0 HPUs
    I0328 23:01:07.645901 139939454809920 rank_zero.py:64] HPU available: False, using: 0 HPUs
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    I0328 23:01:12.623627 139939454809920 cuda.py:61] LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    I0328 23:01:12.632704 139939454809920 model_summary.py:94] 
      | Name      | Type               | Params
    -------------------------------------------------
    0 | model     | GraphModule        | 14.0 M
    1 | loss_fn   | CrossEntropyLoss   | 0     
    2 | acc_train | MulticlassAccuracy | 0     
    3 | loss_val  | MeanMetric         | 0     
    4 | loss_test | MeanMetric         | 0     
    -------------------------------------------------
    14.0 M    Trainable params
    0         Non-trainable params
    14.0 M    Total params
    56.118    Total estimated model params size (MB)
    Epoch 0: 100%|█| 782/782 [00:36<00:00, 21.71it/s, v_num=14, train_acc_step=0.938
    Validation: |                                             | 0/? [00:00<?, ?it/s][A
    Validation:   0%|                                       | 0/157 [00:00<?, ?it/s][A
    Validation DataLoader 0:   0%|                          | 0/157 [00:00<?, ?it/s][A
    Validation DataLoader 0:   1%|                  | 1/157 [00:00<00:03, 48.47it/s][A
    Validation DataLoader 0:   1%|▏                 | 2/157 [00:00<00:03, 49.82it/s][A
    Validation DataLoader 0:   2%|▎                 | 3/157 [00:00<00:03, 50.58it/s][A
    Validation DataLoader 0:   3%|▍                 | 4/157 [00:00<00:03, 49.84it/s][A
    Validation DataLoader 0:   3%|▌                 | 5/157 [00:00<00:03, 50.45it/s][A
    Validation DataLoader 0:   4%|▋                 | 6/157 [00:00<00:02, 50.79it/s][A
    Validation DataLoader 0:   4%|▊                 | 7/157 [00:00<00:02, 51.03it/s][A
    Validation DataLoader 0:   5%|▉                 | 8/157 [00:00<00:02, 51.28it/s][A
    Validation DataLoader 0:   6%|█                 | 9/157 [00:00<00:02, 51.86it/s][A
    Validation DataLoader 0:   6%|█                | 10/157 [00:00<00:02, 52.48it/s][A
    Validation DataLoader 0:   7%|█▏               | 11/157 [00:00<00:02, 51.76it/s][A
    Validation DataLoader 0:   8%|█▎               | 12/157 [00:00<00:02, 52.32it/s][A
    Validation DataLoader 0:   8%|█▍               | 13/157 [00:00<00:02, 51.83it/s][A
    Validation DataLoader 0:   9%|█▌               | 14/157 [00:00<00:02, 51.22it/s][A
    Validation DataLoader 0:  10%|█▌               | 15/157 [00:00<00:02, 51.72it/s][A
    Validation DataLoader 0:  10%|█▋               | 16/157 [00:00<00:02, 52.16it/s][A
    Validation DataLoader 0:  11%|█▊               | 17/157 [00:00<00:02, 51.60it/s][A
    Validation DataLoader 0:  11%|█▉               | 18/157 [00:00<00:02, 50.72it/s][A
    Validation DataLoader 0:  12%|██               | 19/157 [00:00<00:02, 51.01it/s][A
    Validation DataLoader 0:  13%|██▏              | 20/157 [00:00<00:02, 51.09it/s][A
    Validation DataLoader 0:  13%|██▎              | 21/157 [00:00<00:02, 51.49it/s][A
    Validation DataLoader 0:  14%|██▍              | 22/157 [00:00<00:02, 51.40it/s][A
    Validation DataLoader 0:  15%|██▍              | 23/157 [00:00<00:02, 51.65it/s][A
    Validation DataLoader 0:  15%|██▌              | 24/157 [00:00<00:02, 51.76it/s][A
    Validation DataLoader 0:  16%|██▋              | 25/157 [00:00<00:02, 52.09it/s][A
    Validation DataLoader 0:  17%|██▊              | 26/157 [00:00<00:02, 52.40it/s][A
    Validation DataLoader 0:  17%|██▉              | 27/157 [00:00<00:02, 52.68it/s][A
    Validation DataLoader 0:  18%|███              | 28/157 [00:00<00:02, 52.53it/s][A
    Validation DataLoader 0:  18%|███▏             | 29/157 [00:00<00:02, 52.67it/s][A
    Validation DataLoader 0:  19%|███▏             | 30/157 [00:00<00:02, 52.92it/s][A
    Validation DataLoader 0:  20%|███▎             | 31/157 [00:00<00:02, 53.15it/s][A
    Validation DataLoader 0:  20%|███▍             | 32/157 [00:00<00:02, 53.37it/s][A
    Validation DataLoader 0:  21%|███▌             | 33/157 [00:00<00:02, 53.58it/s][A
    Validation DataLoader 0:  22%|███▋             | 34/157 [00:00<00:02, 53.77it/s][A
    Validation DataLoader 0:  22%|███▊             | 35/157 [00:00<00:02, 53.96it/s][A
    Validation DataLoader 0:  23%|███▉             | 36/157 [00:00<00:02, 54.12it/s][A
    Validation DataLoader 0:  24%|████             | 37/157 [00:00<00:02, 54.28it/s][A
    Validation DataLoader 0:  24%|████             | 38/157 [00:00<00:02, 54.42it/s][A
    Validation DataLoader 0:  25%|████▏            | 39/157 [00:00<00:02, 54.56it/s][A
    Validation DataLoader 0:  25%|████▎            | 40/157 [00:00<00:02, 54.68it/s][A
    Validation DataLoader 0:  26%|████▍            | 41/157 [00:00<00:02, 54.81it/s][A
    Validation DataLoader 0:  27%|████▌            | 42/157 [00:00<00:02, 54.94it/s][A
    Validation DataLoader 0:  27%|████▋            | 43/157 [00:00<00:02, 55.06it/s][A
    Validation DataLoader 0:  28%|████▊            | 44/157 [00:00<00:02, 55.18it/s][A
    Validation DataLoader 0:  29%|████▊            | 45/157 [00:00<00:02, 55.30it/s][A
    Validation DataLoader 0:  29%|████▉            | 46/157 [00:00<00:02, 55.40it/s][A
    Validation DataLoader 0:  30%|█████            | 47/157 [00:00<00:01, 55.50it/s][A
    Validation DataLoader 0:  31%|█████▏           | 48/157 [00:00<00:01, 55.60it/s][A
    Validation DataLoader 0:  31%|█████▎           | 49/157 [00:00<00:01, 55.68it/s][A
    Validation DataLoader 0:  32%|█████▍           | 50/157 [00:00<00:01, 55.65it/s][A
    Validation DataLoader 0:  32%|█████▌           | 51/157 [00:00<00:01, 55.74it/s][A
    Validation DataLoader 0:  33%|█████▋           | 52/157 [00:00<00:01, 55.78it/s][A
    Validation DataLoader 0:  34%|█████▋           | 53/157 [00:00<00:01, 55.86it/s][A
    Validation DataLoader 0:  34%|█████▊           | 54/157 [00:00<00:01, 55.95it/s][A
    Validation DataLoader 0:  35%|█████▉           | 55/157 [00:00<00:01, 55.98it/s][A
    Validation DataLoader 0:  36%|██████           | 56/157 [00:01<00:01, 55.90it/s][A
    Validation DataLoader 0:  36%|██████▏          | 57/157 [00:01<00:01, 55.88it/s][A
    Validation DataLoader 0:  37%|██████▎          | 58/157 [00:01<00:01, 55.91it/s][A
    Validation DataLoader 0:  38%|██████▍          | 59/157 [00:01<00:01, 55.95it/s][A
    Validation DataLoader 0:  38%|██████▍          | 60/157 [00:01<00:01, 55.99it/s][A
    Validation DataLoader 0:  39%|██████▌          | 61/157 [00:01<00:01, 56.01it/s][A
    Validation DataLoader 0:  39%|██████▋          | 62/157 [00:01<00:01, 56.04it/s][A
    Validation DataLoader 0:  40%|██████▊          | 63/157 [00:01<00:01, 56.06it/s][A
    Validation DataLoader 0:  41%|██████▉          | 64/157 [00:01<00:01, 56.08it/s][A
    Validation DataLoader 0:  41%|███████          | 65/157 [00:01<00:01, 56.10it/s][A
    Validation DataLoader 0:  42%|███████▏         | 66/157 [00:01<00:01, 56.14it/s][A
    Validation DataLoader 0:  43%|███████▎         | 67/157 [00:01<00:01, 56.16it/s][A
    Validation DataLoader 0:  43%|███████▎         | 68/157 [00:01<00:01, 56.17it/s][A
    Validation DataLoader 0:  44%|███████▍         | 69/157 [00:01<00:01, 56.20it/s][A
    Validation DataLoader 0:  45%|███████▌         | 70/157 [00:01<00:01, 56.22it/s][A
    Validation DataLoader 0:  45%|███████▋         | 71/157 [00:01<00:01, 56.24it/s][A
    Validation DataLoader 0:  46%|███████▊         | 72/157 [00:01<00:01, 56.25it/s][A
    Validation DataLoader 0:  46%|███████▉         | 73/157 [00:01<00:01, 56.27it/s][A
    Validation DataLoader 0:  47%|████████         | 74/157 [00:01<00:01, 56.29it/s][A
    Validation DataLoader 0:  48%|████████         | 75/157 [00:01<00:01, 56.31it/s][A
    Validation DataLoader 0:  48%|████████▏        | 76/157 [00:01<00:01, 56.32it/s][A
    Validation DataLoader 0:  49%|████████▎        | 77/157 [00:01<00:01, 56.33it/s][A
    Validation DataLoader 0:  50%|████████▍        | 78/157 [00:01<00:01, 56.36it/s][A
    Validation DataLoader 0:  50%|████████▌        | 79/157 [00:01<00:01, 56.37it/s][A
    Validation DataLoader 0:  51%|████████▋        | 80/157 [00:01<00:01, 56.39it/s][A
    Validation DataLoader 0:  52%|████████▊        | 81/157 [00:01<00:01, 56.41it/s][A
    Validation DataLoader 0:  52%|████████▉        | 82/157 [00:01<00:01, 56.42it/s][A
    Validation DataLoader 0:  53%|████████▉        | 83/157 [00:01<00:01, 56.43it/s][A
    Validation DataLoader 0:  54%|█████████        | 84/157 [00:01<00:01, 56.33it/s][A
    Validation DataLoader 0:  54%|█████████▏       | 85/157 [00:01<00:01, 56.31it/s][A
    Validation DataLoader 0:  55%|█████████▎       | 86/157 [00:01<00:01, 56.33it/s][A
    Validation DataLoader 0:  55%|█████████▍       | 87/157 [00:01<00:01, 56.35it/s][A
    Validation DataLoader 0:  56%|█████████▌       | 88/157 [00:01<00:01, 56.37it/s][A
    Validation DataLoader 0:  57%|█████████▋       | 89/157 [00:01<00:01, 56.40it/s][A
    Validation DataLoader 0:  57%|█████████▋       | 90/157 [00:01<00:01, 56.41it/s][A
    Validation DataLoader 0:  58%|█████████▊       | 91/157 [00:01<00:01, 56.43it/s][A
    Validation DataLoader 0:  59%|█████████▉       | 92/157 [00:01<00:01, 56.44it/s][A
    Validation DataLoader 0:  59%|██████████       | 93/157 [00:01<00:01, 56.45it/s][A
    Validation DataLoader 0:  60%|██████████▏      | 94/157 [00:01<00:01, 56.46it/s][A
    Validation DataLoader 0:  61%|██████████▎      | 95/157 [00:01<00:01, 56.47it/s][A
    Validation DataLoader 0:  61%|██████████▍      | 96/157 [00:01<00:01, 56.48it/s][A
    Validation DataLoader 0:  62%|██████████▌      | 97/157 [00:01<00:01, 56.49it/s][A
    Validation DataLoader 0:  62%|██████████▌      | 98/157 [00:01<00:01, 56.50it/s][A
    Validation DataLoader 0:  63%|██████████▋      | 99/157 [00:01<00:01, 56.51it/s][A
    Validation DataLoader 0:  64%|██████████▏     | 100/157 [00:01<00:01, 56.51it/s][A
    Validation DataLoader 0:  64%|██████████▎     | 101/157 [00:01<00:00, 56.52it/s][A
    Validation DataLoader 0:  65%|██████████▍     | 102/157 [00:01<00:00, 56.53it/s][A
    Validation DataLoader 0:  66%|██████████▍     | 103/157 [00:01<00:00, 56.55it/s][A
    Validation DataLoader 0:  66%|██████████▌     | 104/157 [00:01<00:00, 56.57it/s][A
    Validation DataLoader 0:  67%|██████████▋     | 105/157 [00:01<00:00, 56.58it/s][A
    Validation DataLoader 0:  68%|██████████▊     | 106/157 [00:01<00:00, 56.60it/s][A
    Validation DataLoader 0:  68%|██████████▉     | 107/157 [00:01<00:00, 56.61it/s][A
    Validation DataLoader 0:  69%|███████████     | 108/157 [00:01<00:00, 56.62it/s][A
    Validation DataLoader 0:  69%|███████████     | 109/157 [00:01<00:00, 56.64it/s][A
    Validation DataLoader 0:  70%|███████████▏    | 110/157 [00:01<00:00, 56.65it/s][A
    Validation DataLoader 0:  71%|███████████▎    | 111/157 [00:01<00:00, 56.66it/s][A
    Validation DataLoader 0:  71%|███████████▍    | 112/157 [00:01<00:00, 56.67it/s][A
    Validation DataLoader 0:  72%|███████████▌    | 113/157 [00:01<00:00, 56.69it/s][A
    Validation DataLoader 0:  73%|███████████▌    | 114/157 [00:02<00:00, 56.69it/s][A
    Validation DataLoader 0:  73%|███████████▋    | 115/157 [00:02<00:00, 56.70it/s][A
    Validation DataLoader 0:  74%|███████████▊    | 116/157 [00:02<00:00, 56.71it/s][A
    Validation DataLoader 0:  75%|███████████▉    | 117/157 [00:02<00:00, 56.73it/s][A
    Validation DataLoader 0:  75%|████████████    | 118/157 [00:02<00:00, 56.74it/s][A
    Validation DataLoader 0:  76%|████████████▏   | 119/157 [00:02<00:00, 56.75it/s][A
    Validation DataLoader 0:  76%|████████████▏   | 120/157 [00:02<00:00, 56.76it/s][A
    Validation DataLoader 0:  77%|████████████▎   | 121/157 [00:02<00:00, 56.76it/s][A
    Validation DataLoader 0:  78%|████████████▍   | 122/157 [00:02<00:00, 56.77it/s][A
    Validation DataLoader 0:  78%|████████████▌   | 123/157 [00:02<00:00, 56.77it/s][A
    Validation DataLoader 0:  79%|████████████▋   | 124/157 [00:02<00:00, 56.78it/s][A
    Validation DataLoader 0:  80%|████████████▋   | 125/157 [00:02<00:00, 56.80it/s][A
    Validation DataLoader 0:  80%|████████████▊   | 126/157 [00:02<00:00, 56.81it/s][A
    Validation DataLoader 0:  81%|████████████▉   | 127/157 [00:02<00:00, 56.81it/s][A
    Validation DataLoader 0:  82%|█████████████   | 128/157 [00:02<00:00, 56.82it/s][A
    Validation DataLoader 0:  82%|█████████████▏  | 129/157 [00:02<00:00, 56.82it/s][A
    Validation DataLoader 0:  83%|█████████████▏  | 130/157 [00:02<00:00, 56.83it/s][A
    Validation DataLoader 0:  83%|█████████████▎  | 131/157 [00:02<00:00, 56.84it/s][A
    Validation DataLoader 0:  84%|█████████████▍  | 132/157 [00:02<00:00, 56.84it/s][A
    Validation DataLoader 0:  85%|█████████████▌  | 133/157 [00:02<00:00, 56.85it/s][A
    Validation DataLoader 0:  85%|█████████████▋  | 134/157 [00:02<00:00, 56.86it/s][A
    Validation DataLoader 0:  86%|█████████████▊  | 135/157 [00:02<00:00, 56.87it/s][A
    Validation DataLoader 0:  87%|█████████████▊  | 136/157 [00:02<00:00, 56.87it/s][A
    Validation DataLoader 0:  87%|█████████████▉  | 137/157 [00:02<00:00, 56.87it/s][A
    Validation DataLoader 0:  88%|██████████████  | 138/157 [00:02<00:00, 56.88it/s][A
    Validation DataLoader 0:  89%|██████████████▏ | 139/157 [00:02<00:00, 56.89it/s][A
    Validation DataLoader 0:  89%|██████████████▎ | 140/157 [00:02<00:00, 56.89it/s][A
    Validation DataLoader 0:  90%|██████████████▎ | 141/157 [00:02<00:00, 56.89it/s][A
    Validation DataLoader 0:  90%|██████████████▍ | 142/157 [00:02<00:00, 56.90it/s][A
    Validation DataLoader 0:  91%|██████████████▌ | 143/157 [00:02<00:00, 56.90it/s][A
    Validation DataLoader 0:  92%|██████████████▋ | 144/157 [00:02<00:00, 56.91it/s][A
    Validation DataLoader 0:  92%|██████████████▊ | 145/157 [00:02<00:00, 56.92it/s][A
    Validation DataLoader 0:  93%|██████████████▉ | 146/157 [00:02<00:00, 56.93it/s][A
    Validation DataLoader 0:  94%|██████████████▉ | 147/157 [00:02<00:00, 56.94it/s][A
    Validation DataLoader 0:  94%|███████████████ | 148/157 [00:02<00:00, 56.95it/s][A
    Validation DataLoader 0:  95%|███████████████▏| 149/157 [00:02<00:00, 56.96it/s][A
    Validation DataLoader 0:  96%|███████████████▎| 150/157 [00:02<00:00, 56.97it/s][A
    Validation DataLoader 0:  96%|███████████████▍| 151/157 [00:02<00:00, 56.98it/s][A
    Validation DataLoader 0:  97%|███████████████▍| 152/157 [00:02<00:00, 56.98it/s][A
    Validation DataLoader 0:  97%|███████████████▌| 153/157 [00:02<00:00, 56.98it/s][A
    Validation DataLoader 0:  98%|███████████████▋| 154/157 [00:02<00:00, 56.99it/s][A
    Validation DataLoader 0:  99%|███████████████▊| 155/157 [00:02<00:00, 56.99it/s][A
    Validation DataLoader 0:  99%|███████████████▉| 156/157 [00:02<00:00, 56.97it/s][A
    Validation DataLoader 0: 100%|████████████████| 157/157 [00:02<00:00, 57.15it/s][A
    Epoch 1: 100%|█| 782/782 [00:43<00:00, 18.17it/s, v_num=14, train_acc_step=0.812[A
    Validation: |                                             | 0/? [00:00<?, ?it/s][A
    Validation:   0%|                                       | 0/157 [00:00<?, ?it/s][A
    Validation DataLoader 0:   0%|                          | 0/157 [00:00<?, ?it/s][A
    Validation DataLoader 0:   1%|                  | 1/157 [00:00<00:08, 19.41it/s][A
    Validation DataLoader 0:   1%|▏                 | 2/157 [00:00<00:05, 27.82it/s][A
    Validation DataLoader 0:   2%|▎                 | 3/157 [00:00<00:05, 30.69it/s][A
    Validation DataLoader 0:   3%|▍                 | 4/157 [00:00<00:04, 33.41it/s][A
    Validation DataLoader 0:   3%|▌                 | 5/157 [00:00<00:04, 35.97it/s][A
    Validation DataLoader 0:   4%|▋                 | 6/157 [00:00<00:03, 37.98it/s][A
    Validation DataLoader 0:   4%|▊                 | 7/157 [00:00<00:03, 39.56it/s][A
    Validation DataLoader 0:   5%|▉                 | 8/157 [00:00<00:03, 37.82it/s][A
    Validation DataLoader 0:   6%|█                 | 9/157 [00:00<00:03, 39.03it/s][A
    Validation DataLoader 0:   6%|█                | 10/157 [00:00<00:03, 36.79it/s][A
    Validation DataLoader 0:   7%|█▏               | 11/157 [00:00<00:03, 37.70it/s][A
    Validation DataLoader 0:   8%|█▎               | 12/157 [00:00<00:03, 38.78it/s][A
    Validation DataLoader 0:   8%|█▍               | 13/157 [00:00<00:03, 39.57it/s][A
    Validation DataLoader 0:   9%|█▌               | 14/157 [00:00<00:03, 40.44it/s][A
    Validation DataLoader 0:  10%|█▌               | 15/157 [00:00<00:03, 41.38it/s][A
    Validation DataLoader 0:  10%|█▋               | 16/157 [00:00<00:03, 42.21it/s][A
    Validation DataLoader 0:  11%|█▊               | 17/157 [00:00<00:03, 42.51it/s][A
    Validation DataLoader 0:  11%|█▉               | 18/157 [00:00<00:03, 42.74it/s][A
    Validation DataLoader 0:  12%|██               | 19/157 [00:00<00:03, 43.44it/s][A
    Validation DataLoader 0:  13%|██▏              | 20/157 [00:00<00:03, 44.09it/s][A
    Validation DataLoader 0:  13%|██▎              | 21/157 [00:00<00:03, 44.54it/s][A
    Validation DataLoader 0:  14%|██▍              | 22/157 [00:00<00:02, 45.10it/s][A
    Validation DataLoader 0:  15%|██▍              | 23/157 [00:00<00:02, 45.08it/s][A
    Validation DataLoader 0:  15%|██▌              | 24/157 [00:00<00:02, 45.59it/s][A
    Validation DataLoader 0:  16%|██▋              | 25/157 [00:00<00:02, 46.05it/s][A
    Validation DataLoader 0:  17%|██▊              | 26/157 [00:00<00:02, 46.49it/s][A
    Validation DataLoader 0:  17%|██▉              | 27/157 [00:00<00:02, 46.89it/s][A
    Validation DataLoader 0:  18%|███              | 28/157 [00:00<00:02, 47.27it/s][A
    Validation DataLoader 0:  18%|███▏             | 29/157 [00:00<00:02, 47.64it/s][A
    Validation DataLoader 0:  19%|███▏             | 30/157 [00:00<00:02, 46.89it/s][A
    Validation DataLoader 0:  20%|███▎             | 31/157 [00:00<00:02, 47.24it/s][A
    Validation DataLoader 0:  20%|███▍             | 32/157 [00:00<00:02, 47.56it/s][A
    Validation DataLoader 0:  21%|███▌             | 33/157 [00:00<00:02, 44.02it/s][A
    Validation DataLoader 0:  22%|███▋             | 34/157 [00:00<00:02, 44.31it/s][A
    Validation DataLoader 0:  22%|███▊             | 35/157 [00:00<00:02, 44.65it/s][A
    Validation DataLoader 0:  23%|███▉             | 36/157 [00:00<00:02, 42.92it/s][A
    Validation DataLoader 0:  24%|████             | 37/157 [00:00<00:02, 43.21it/s][A
    Validation DataLoader 0:  24%|████             | 38/157 [00:00<00:02, 43.55it/s][A
    Validation DataLoader 0:  25%|████▏            | 39/157 [00:00<00:02, 43.89it/s][A
    Validation DataLoader 0:  25%|████▎            | 40/157 [00:00<00:02, 44.20it/s][A
    Validation DataLoader 0:  26%|████▍            | 41/157 [00:00<00:02, 44.37it/s][A
    Validation DataLoader 0:  27%|████▌            | 42/157 [00:00<00:02, 44.61it/s][A
    Validation DataLoader 0:  27%|████▋            | 43/157 [00:00<00:02, 44.90it/s][A
    Validation DataLoader 0:  28%|████▊            | 44/157 [00:00<00:02, 45.17it/s][A
    Validation DataLoader 0:  29%|████▊            | 45/157 [00:00<00:02, 45.43it/s][A
    Validation DataLoader 0:  29%|████▉            | 46/157 [00:01<00:02, 45.69it/s][A
    Validation DataLoader 0:  30%|█████            | 47/157 [00:01<00:02, 45.93it/s][A
    Validation DataLoader 0:  31%|█████▏           | 48/157 [00:01<00:02, 46.17it/s][A
    Validation DataLoader 0:  31%|█████▎           | 49/157 [00:01<00:02, 46.40it/s][A
    Validation DataLoader 0:  32%|█████▍           | 50/157 [00:01<00:02, 46.62it/s][A
    Validation DataLoader 0:  32%|█████▌           | 51/157 [00:01<00:02, 46.80it/s][A
    Validation DataLoader 0:  33%|█████▋           | 52/157 [00:01<00:02, 47.01it/s][A
    Validation DataLoader 0:  34%|█████▋           | 53/157 [00:01<00:02, 47.21it/s][A
    Validation DataLoader 0:  34%|█████▊           | 54/157 [00:01<00:02, 47.41it/s][A
    Validation DataLoader 0:  35%|█████▉           | 55/157 [00:01<00:02, 47.60it/s][A
    Validation DataLoader 0:  36%|██████           | 56/157 [00:01<00:02, 47.78it/s][A
    Validation DataLoader 0:  36%|██████▏          | 57/157 [00:01<00:02, 47.92it/s][A
    Validation DataLoader 0:  37%|██████▎          | 58/157 [00:01<00:02, 48.05it/s][A
    Validation DataLoader 0:  38%|██████▍          | 59/157 [00:01<00:02, 48.18it/s][A
    Validation DataLoader 0:  38%|██████▍          | 60/157 [00:01<00:02, 48.31it/s][A
    Validation DataLoader 0:  39%|██████▌          | 61/157 [00:01<00:01, 48.34it/s][A
    Validation DataLoader 0:  39%|██████▋          | 62/157 [00:01<00:01, 48.43it/s][A
    Validation DataLoader 0:  40%|██████▊          | 63/157 [00:01<00:01, 48.56it/s][A
    Validation DataLoader 0:  41%|██████▉          | 64/157 [00:01<00:01, 48.58it/s][A
    Validation DataLoader 0:  41%|███████          | 65/157 [00:01<00:01, 48.67it/s][A
    Validation DataLoader 0:  42%|███████▏         | 66/157 [00:01<00:01, 48.79it/s][A
    Validation DataLoader 0:  43%|███████▎         | 67/157 [00:01<00:01, 48.91it/s][A
    Validation DataLoader 0:  43%|███████▎         | 68/157 [00:01<00:01, 49.02it/s][A
    Validation DataLoader 0:  44%|███████▍         | 69/157 [00:01<00:01, 49.03it/s][A
    Validation DataLoader 0:  45%|███████▌         | 70/157 [00:01<00:01, 49.10it/s][A
    Validation DataLoader 0:  45%|███████▋         | 71/157 [00:01<00:01, 49.20it/s][A
    Validation DataLoader 0:  46%|███████▊         | 72/157 [00:01<00:01, 49.31it/s][A
    Validation DataLoader 0:  46%|███████▉         | 73/157 [00:01<00:01, 49.41it/s][A
    Validation DataLoader 0:  47%|████████         | 74/157 [00:01<00:01, 49.51it/s][A
    Validation DataLoader 0:  48%|████████         | 75/157 [00:01<00:01, 49.61it/s][A
    Validation DataLoader 0:  48%|████████▏        | 76/157 [00:01<00:01, 49.70it/s][A
    Validation DataLoader 0:  49%|████████▎        | 77/157 [00:01<00:01, 49.79it/s][A
    Validation DataLoader 0:  50%|████████▍        | 78/157 [00:01<00:01, 49.87it/s][A
    Validation DataLoader 0:  50%|████████▌        | 79/157 [00:01<00:01, 49.96it/s][A
    Validation DataLoader 0:  51%|████████▋        | 80/157 [00:01<00:01, 50.05it/s][A
    Validation DataLoader 0:  52%|████████▊        | 81/157 [00:01<00:01, 50.13it/s][A
    Validation DataLoader 0:  52%|████████▉        | 82/157 [00:01<00:01, 50.22it/s][A
    Validation DataLoader 0:  53%|████████▉        | 83/157 [00:01<00:01, 50.29it/s][A
    Validation DataLoader 0:  54%|█████████        | 84/157 [00:01<00:01, 50.37it/s][A
    Validation DataLoader 0:  54%|█████████▏       | 85/157 [00:01<00:01, 50.45it/s][A
    Validation DataLoader 0:  55%|█████████▎       | 86/157 [00:01<00:01, 50.52it/s][A
    Validation DataLoader 0:  55%|█████████▍       | 87/157 [00:01<00:01, 50.60it/s][A
    Validation DataLoader 0:  56%|█████████▌       | 88/157 [00:01<00:01, 50.67it/s][A
    Validation DataLoader 0:  57%|█████████▋       | 89/157 [00:01<00:01, 50.67it/s][A
    Validation DataLoader 0:  57%|█████████▋       | 90/157 [00:01<00:01, 50.70it/s][A
    Validation DataLoader 0:  58%|█████████▊       | 91/157 [00:01<00:01, 50.77it/s][A
    Validation DataLoader 0:  59%|█████████▉       | 92/157 [00:01<00:01, 50.76it/s][A
    Validation DataLoader 0:  59%|██████████       | 93/157 [00:01<00:01, 50.80it/s][A
    Validation DataLoader 0:  60%|██████████▏      | 94/157 [00:01<00:01, 50.87it/s][A
    Validation DataLoader 0:  61%|██████████▎      | 95/157 [00:01<00:01, 50.94it/s][A
    Validation DataLoader 0:  61%|██████████▍      | 96/157 [00:01<00:01, 51.00it/s][A
    Validation DataLoader 0:  62%|██████████▌      | 97/157 [00:01<00:01, 50.99it/s][A
    Validation DataLoader 0:  62%|██████████▌      | 98/157 [00:01<00:01, 51.02it/s][A
    Validation DataLoader 0:  63%|██████████▋      | 99/157 [00:01<00:01, 51.08it/s][A
    Validation DataLoader 0:  64%|██████████▏     | 100/157 [00:01<00:01, 51.15it/s][A
    Validation DataLoader 0:  64%|██████████▎     | 101/157 [00:01<00:01, 51.21it/s][A
    Validation DataLoader 0:  65%|██████████▍     | 102/157 [00:01<00:01, 51.27it/s][A
    Validation DataLoader 0:  66%|██████████▍     | 103/157 [00:02<00:01, 51.33it/s][A
    Validation DataLoader 0:  66%|██████████▌     | 104/157 [00:02<00:01, 51.39it/s][A
    Validation DataLoader 0:  67%|██████████▋     | 105/157 [00:02<00:01, 51.45it/s][A
    Validation DataLoader 0:  68%|██████████▊     | 106/157 [00:02<00:00, 51.51it/s][A
    Validation DataLoader 0:  68%|██████████▉     | 107/157 [00:02<00:00, 51.57it/s][A
    Validation DataLoader 0:  69%|███████████     | 108/157 [00:02<00:00, 51.63it/s][A
    Validation DataLoader 0:  69%|███████████     | 109/157 [00:02<00:00, 51.68it/s][A
    Validation DataLoader 0:  70%|███████████▏    | 110/157 [00:02<00:00, 51.73it/s][A
    Validation DataLoader 0:  71%|███████████▎    | 111/157 [00:02<00:00, 51.79it/s][A
    Validation DataLoader 0:  71%|███████████▍    | 112/157 [00:02<00:00, 51.84it/s][A
    Validation DataLoader 0:  72%|███████████▌    | 113/157 [00:02<00:00, 51.89it/s][A
    Validation DataLoader 0:  73%|███████████▌    | 114/157 [00:02<00:00, 51.94it/s][A
    Validation DataLoader 0:  73%|███████████▋    | 115/157 [00:02<00:00, 51.99it/s][A
    Validation DataLoader 0:  74%|███████████▊    | 116/157 [00:02<00:00, 52.04it/s][A
    Validation DataLoader 0:  75%|███████████▉    | 117/157 [00:02<00:00, 52.09it/s][A
    Validation DataLoader 0:  75%|████████████    | 118/157 [00:02<00:00, 52.13it/s][A
    Validation DataLoader 0:  76%|████████████▏   | 119/157 [00:02<00:00, 52.18it/s][A
    Validation DataLoader 0:  76%|████████████▏   | 120/157 [00:02<00:00, 52.23it/s][A
    Validation DataLoader 0:  77%|████████████▎   | 121/157 [00:02<00:00, 52.27it/s][A
    Validation DataLoader 0:  78%|████████████▍   | 122/157 [00:02<00:00, 52.32it/s][A
    Validation DataLoader 0:  78%|████████████▌   | 123/157 [00:02<00:00, 52.36it/s][A
    Validation DataLoader 0:  79%|████████████▋   | 124/157 [00:02<00:00, 52.41it/s][A
    Validation DataLoader 0:  80%|████████████▋   | 125/157 [00:02<00:00, 52.45it/s][A
    Validation DataLoader 0:  80%|████████████▊   | 126/157 [00:02<00:00, 52.50it/s][A
    Validation DataLoader 0:  81%|████████████▉   | 127/157 [00:02<00:00, 52.54it/s][A
    Validation DataLoader 0:  82%|█████████████   | 128/157 [00:02<00:00, 52.58it/s][A
    Validation DataLoader 0:  82%|█████████████▏  | 129/157 [00:02<00:00, 52.62it/s][A
    Validation DataLoader 0:  83%|█████████████▏  | 130/157 [00:02<00:00, 52.67it/s][A
    Validation DataLoader 0:  83%|█████████████▎  | 131/157 [00:02<00:00, 52.70it/s][A
    Validation DataLoader 0:  84%|█████████████▍  | 132/157 [00:02<00:00, 52.74it/s][A
    Validation DataLoader 0:  85%|█████████████▌  | 133/157 [00:02<00:00, 52.78it/s][A
    Validation DataLoader 0:  85%|█████████████▋  | 134/157 [00:02<00:00, 52.81it/s][A
    Validation DataLoader 0:  86%|█████████████▊  | 135/157 [00:02<00:00, 52.85it/s][A
    Validation DataLoader 0:  87%|█████████████▊  | 136/157 [00:02<00:00, 52.89it/s][A
    Validation DataLoader 0:  87%|█████████████▉  | 137/157 [00:02<00:00, 52.93it/s][A
    Validation DataLoader 0:  88%|██████████████  | 138/157 [00:02<00:00, 52.96it/s][A
    Validation DataLoader 0:  89%|██████████████▏ | 139/157 [00:02<00:00, 53.00it/s][A
    Validation DataLoader 0:  89%|██████████████▎ | 140/157 [00:02<00:00, 53.03it/s][A
    Validation DataLoader 0:  90%|██████████████▎ | 141/157 [00:02<00:00, 53.06it/s][A
    Validation DataLoader 0:  90%|██████████████▍ | 142/157 [00:02<00:00, 53.10it/s][A
    Validation DataLoader 0:  91%|██████████████▌ | 143/157 [00:02<00:00, 53.13it/s][A
    Validation DataLoader 0:  92%|██████████████▋ | 144/157 [00:02<00:00, 53.16it/s][A
    Validation DataLoader 0:  92%|██████████████▊ | 145/157 [00:02<00:00, 53.19it/s][A
    Validation DataLoader 0:  93%|██████████████▉ | 146/157 [00:02<00:00, 53.22it/s][A
    Validation DataLoader 0:  94%|██████████████▉ | 147/157 [00:02<00:00, 53.26it/s][A
    Validation DataLoader 0:  94%|███████████████ | 148/157 [00:02<00:00, 53.29it/s][A
    Validation DataLoader 0:  95%|███████████████▏| 149/157 [00:02<00:00, 53.32it/s][A
    Validation DataLoader 0:  96%|███████████████▎| 150/157 [00:02<00:00, 53.35it/s][A
    Validation DataLoader 0:  96%|███████████████▍| 151/157 [00:02<00:00, 53.39it/s][A
    Validation DataLoader 0:  97%|███████████████▍| 152/157 [00:02<00:00, 53.41it/s][A
    Validation DataLoader 0:  97%|███████████████▌| 153/157 [00:02<00:00, 53.44it/s][A
    Validation DataLoader 0:  98%|███████████████▋| 154/157 [00:02<00:00, 53.47it/s][A
    Validation DataLoader 0:  99%|███████████████▊| 155/157 [00:02<00:00, 53.50it/s][A
    Validation DataLoader 0:  99%|███████████████▉| 156/157 [00:02<00:00, 53.53it/s][A
    Validation DataLoader 0: 100%|████████████████| 157/157 [00:02<00:00, 53.69it/s][A
    Epoch 1: 100%|█| 782/782 [00:54<00:00, 14.27it/s, v_num=14, train_acc_step=0.812[AINFO: `Trainer.fit` stopped: `max_epochs=2` reached.
    I0328 23:03:10.943096 139939454809920 rank_zero.py:64] `Trainer.fit` stopped: `max_epochs=2` reached.
    Epoch 1: 100%|█| 782/782 [00:55<00:00, 14.01it/s, v_num=14, train_acc_step=0.812
    [32mINFO    [0m [34mFine Tuning Complete[0m
    I0328 23:03:18.680018 139939454809920 fine_tune.py:161] Fine Tuning Complete
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0328 23:03:18.687829 139939454809920 quantize.py:209] Converting PyTorch model to ONNX...
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:363: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax < 0:
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:366: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      max_bound = torch.tensor((2.0**(num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:376: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:382: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax <= epsilon:
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_39/model.onnx[0m
    I0328 23:03:28.297990 139939454809920 quantize.py:239] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_39/model.onnx
    [32mINFO    [0m [34mConverting PyTorch model to TensorRT...[0m
    I0328 23:03:28.300960 139939454809920 quantize.py:102] Converting PyTorch model to TensorRT...
    [03/28/2024-23:03:36] [TRT] [W] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
    [32mINFO    [0m [34mTensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_40/model.trt[0m
    I0328 23:06:32.223787 139939454809920 quantize.py:202] TensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_40/model.trt
    [32mINFO    [0m [34mTensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_41/model.json[0m
    I0328 23:06:32.581682 139939454809920 quantize.py:259] TensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_41/model.json
    [32mINFO    [0m [34mStarting transformation analysis on vgg7[0m
    I0328 23:06:32.589668 139939454809920 runtime_analysis.py:357] Starting transformation analysis on vgg7
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+------------+
    |      Metric (Per Batch)      |   Value    |
    +------------------------------+------------+
    | Average Validation Accuracy  |  0.92094   |
    |      Average Precision       |  0.91981   |
    |        Average Recall        |  0.92012   |
    |       Average F1 Score       |  0.91983   |
    |         Average Loss         |  0.23915   |
    |       Average Latency        | 8.9375 ms  |
    |   Average GPU Power Usage    |  58.043 W  |
    | Inference Energy Consumption | 0.1441 mWh |
    +------------------------------+------------+[0m
    I0328 23:06:47.111017 139939454809920 runtime_analysis.py:521] 
    Results vgg7:
    +------------------------------+------------+
    |      Metric (Per Batch)      |   Value    |
    +------------------------------+------------+
    | Average Validation Accuracy  |  0.92094   |
    |      Average Precision       |  0.91981   |
    |        Average Recall        |  0.92012   |
    |       Average F1 Score       |  0.91983   |
    |         Average Loss         |  0.23915   |
    |       Average Latency        | 8.9375 ms  |
    |   Average GPU Power Usage    |  58.043 W  |
    | Inference Energy Consumption | 0.1441 mWh |
    +------------------------------+------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_46/model.json[0m
    I0328 23:06:47.114224 139939454809920 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_46/model.json
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-trt_quantized[0m
    I0328 23:06:47.208054 139939454809920 runtime_analysis.py:357] Starting transformation analysis on vgg7-trt_quantized
    [32mINFO    [0m [34m
    Results vgg7-trt_quantized:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.92348   |
    |      Average Precision       |   0.9251    |
    |        Average Recall        |   0.92436   |
    |       Average F1 Score       |   0.92419   |
    |         Average Loss         |   0.24202   |
    |       Average Latency        |  8.5007 ms  |
    |   Average GPU Power Usage    |  52.687 W   |
    | Inference Energy Consumption | 0.12441 mWh |
    +------------------------------+-------------+[0m
    I0328 23:07:00.676242 139939454809920 runtime_analysis.py:521] 
    Results vgg7-trt_quantized:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.92348   |
    |      Average Precision       |   0.9251    |
    |        Average Recall        |   0.92436   |
    |       Average F1 Score       |   0.92419   |
    |         Average Loss         |   0.24202   |
    |       Average Latency        |  8.5007 ms  |
    |   Average GPU Power Usage    |  52.687 W   |
    | Inference Energy Consumption | 0.12441 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/tensorrt/version_7/model.json[0m
    I0328 23:07:00.677799 139939454809920 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/tensorrt/version_7/model.json


By quantizing all convolutional layers to INT8 and maintaining fp16 precision for the linear layers we see a marginal decrease in latency whilst maintaining a comparable accuracy. By experimenting with precisions on a per type basis, you may find insights that work best for your model. 

## Section 4. Layer-wise Mixed Precision

So far we have strictly quantized either in int8 or fp16. Now, we will show how to conduct layerwise mixed precision using the same `vgg7` model. In this case we will show how for instance, layer 0 and 1 can be set to fp16, while the remaining layers can be int8 quantized. 

For this, we set:
- The `by` parameter to `name`
- The `precision` to 'int8' for `passes.tensorrt.default.config`
- The `precision` to 'fp16' for `passes.tensorrt.feature_layers_0.config and passes.tensorrt.feature_layers_1.config`
- The `precision` to 'int8' for `passes.tensorrt.feature_layers_2.config and passes.tensorrt.feature_layers_3.config` (although this is not necessary since the default is already set to 'int8')


```python
VGG_LAYERWISE_TOML = "../../../machop/configs/tensorrt/vgg7_layerwise_mixed_precision.toml"

!ch transform --config {VGG_LAYERWISE_TOML} --load {VGG_CHECKPOINT_PATH} --load-type pl
```

    [2024-03-28 23:25:51,157] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0328 23:25:54.303634 140449214740288 seed.py:54] Seed set to 0
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | Name                    |        Default         | Config. File |     Manual Override      |        Effective         |
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | task                    |     [38;5;8mclassification[0m     |     cls      |                          |           cls            |
    | load_name               |          [38;5;8mNone[0m          |              | /root/mase/mase_output/v | /root/mase/mase_output/v |
    |                         |                        |              |  gg7-pre-trained/test-   |  gg7-pre-trained/test-   |
    |                         |                        |              |     accu-0.9332.ckpt     |     accu-0.9332.ckpt     |
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
    | model                   |          [38;5;8mNone[0m          |     vgg7     |                          |           vgg7           |
    | dataset                 |          [38;5;8mNone[0m          |   cifar10    |                          |         cifar10          |
    | t_max                   |           20           |              |                          |            20            |
    | eta_min                 |         1e-06          |              |                          |          1e-06           |
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    [32mINFO    [0m [34mInitialising model 'vgg7'...[0m
    I0328 23:25:54.313626 140449214740288 cli.py:846] Initialising model 'vgg7'...
    [32mINFO    [0m [34mInitialising dataset 'cifar10'...[0m
    I0328 23:25:54.417618 140449214740288 cli.py:874] Initialising dataset 'cifar10'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-28[0m
    I0328 23:25:54.418019 140449214740288 cli.py:910] Project will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-28
    [32mINFO    [0m [34mTransforming model 'vgg7'...[0m
    I0328 23:25:54.535444 140449214740288 cli.py:370] Transforming model 'vgg7'...
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0328 23:26:00.655963 140449214740288 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0328 23:26:00.777872 140449214740288 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt
    [32mINFO    [0m [34mApplying fake quantization to PyTorch model...[0m
    I0328 23:26:12.570783 140449214740288 utils.py:314] Applying fake quantization to PyTorch model...
    [32mINFO    [0m [34mFake quantization applied to PyTorch model.[0m
    I0328 23:26:12.921518 140449214740288 utils.py:339] Fake quantization applied to PyTorch model.
    [32mINFO    [0m [34mQuantized graph histogram:[0m
    I0328 23:26:12.940881 140449214740288 summary.py:84] Quantized graph histogram:
    [32mINFO    [0m [34m
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm2d     | batch_norm2d |       6 |         0 |           6 |
    | Conv2d          | conv2d       |       6 |         5 |           1 |
    | Linear          | linear       |       3 |         3 |           0 |
    | MaxPool2d       | max_pool2d   |       3 |         0 |           3 |
    | ReLU            | relu         |       8 |         0 |           8 |
    | output          | output       |       1 |         0 |           1 |
    | view            | view         |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |[0m
    I0328 23:26:12.941653 140449214740288 summary.py:85] 
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm2d     | batch_norm2d |       6 |         0 |           6 |
    | Conv2d          | conv2d       |       6 |         5 |           1 |
    | Linear          | linear       |       3 |         3 |           0 |
    | MaxPool2d       | max_pool2d   |       3 |         0 |           3 |
    | ReLU            | relu         |       8 |         0 |           8 |
    | output          | output       |       1 |         0 |           1 |
    | view            | view         |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |
    [32mINFO    [0m [34mStarting calibration of the model in PyTorch...[0m
    I0328 23:26:12.942434 140449214740288 calibrate.py:143] Starting calibration of the model in PyTorch...
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.952852 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.953087 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.953220 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.953323 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.953427 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.953537 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.953635 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.953727 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.953824 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.953913 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.954009 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.954101 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.954192 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.954295 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.954387 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0328 23:26:12.954478 140449214740288 calibrate.py:152] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.535559 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.535989 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.536076 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.536209 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.536275 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.536394 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.536448 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.536548 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.536605 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.536704 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.536756 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.536853 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.536910 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.537009 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.537059 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.537172 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.537243 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.537340 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.537401 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.537500 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.537567 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.537664 140449214740288 tensor_quantizer.py:174] Disable MaxCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.537712 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.537806 140449214740288 tensor_quantizer.py:174] Disable MaxCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.537872 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.537964 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.538011 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.538102 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.538153 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.538257 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0328 23:26:18.538305 140449214740288 calibrate.py:175] Enabling Quantization and Disabling Calibration
    W0328 23:26:18.538399 140449214740288 tensor_quantizer.py:174] Disable HistogramCalibrator
    W0328 23:26:18.546847 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    W0328 23:26:18.546956 140449214740288 tensor_quantizer.py:239] Call .cuda() if running on GPU after loading calibrated amax.
    [32mINFO    [0m [34mfeature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.2937 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.547093 140449214740288 calibrate.py:131] feature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.2937 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.547413 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2366 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.547523 140449214740288 calibrate.py:131] feature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2366 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.547886 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=1.8330 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.548011 140449214740288 calibrate.py:131] feature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=1.8330 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.548304 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2296 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.548407 140449214740288 calibrate.py:131] feature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2296 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.548746 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.4681 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.548850 140449214740288 calibrate.py:131] feature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.4681 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.549174 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2080 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.549281 140449214740288 calibrate.py:131] feature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2080 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.549598 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.9284 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.549701 140449214740288 calibrate.py:131] feature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.9284 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.550004 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2013 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.550112 140449214740288 calibrate.py:131] feature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2013 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.550404 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.6127 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.550505 140449214740288 calibrate.py:131] feature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.6127 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.550795 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.1879 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.550894 140449214740288 calibrate.py:131] feature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.1879 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.551028 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mclassifier.0._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=11.6545 calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.551139 140449214740288 calibrate.py:131] classifier.0._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=11.6545 calibrator=MaxCalibrator scale=1.0 quant)
    W0328 23:26:18.551264 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([1024, 1]).
    [32mINFO    [0m [34mclassifier.0._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.0158, 0.4703](1024) calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.551505 140449214740288 calibrate.py:131] classifier.0._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.0158, 0.4703](1024) calibrator=MaxCalibrator scale=1.0 quant)
    W0328 23:26:18.551799 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mclassifier.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=9.7654 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.551900 140449214740288 calibrate.py:131] classifier.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=9.7654 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.552192 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mclassifier.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.0590 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.552289 140449214740288 calibrate.py:131] classifier.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.0590 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.552608 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mlast_layer._input_quantizer             : TensorQuantizer(8bit fake per-tensor amax=7.4475 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.552708 140449214740288 calibrate.py:131] last_layer._input_quantizer             : TensorQuantizer(8bit fake per-tensor amax=7.4475 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:18.552994 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mlast_layer._weight_quantizer            : TensorQuantizer(8bit fake per-tensor amax=0.1019 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:18.553097 140449214740288 calibrate.py:131] last_layer._weight_quantizer            : TensorQuantizer(8bit fake per-tensor amax=0.1019 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.0...[0m
    I0328 23:26:18.554053 140449214740288 calibrate.py:105] Performing post calibration analysis for calibrator percentile_99.0...
    [32mINFO    [0m [34mStarting transformation analysis on vgg7[0m
    I0328 23:26:18.554283 140449214740288 runtime_analysis.py:357] Starting transformation analysis on vgg7
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.87663   |
    |      Average Precision       |   0.87531   |
    |        Average Recall        |   0.87386   |
    |       Average F1 Score       |   0.87335   |
    |         Average Loss         |   0.65133   |
    |       Average Latency        |  17.901 ms  |
    |   Average GPU Power Usage    |  57.532 W   |
    | Inference Energy Consumption | 0.28607 mWh |
    +------------------------------+-------------+[0m
    I0328 23:26:29.263397 140449214740288 runtime_analysis.py:521] 
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.87663   |
    |      Average Precision       |   0.87531   |
    |        Average Recall        |   0.87386   |
    |       Average F1 Score       |   0.87335   |
    |         Average Loss         |   0.65133   |
    |       Average Latency        |  17.901 ms  |
    |   Average GPU Power Usage    |  57.532 W   |
    | Inference Energy Consumption | 0.28607 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_50/model.json[0m
    I0328 23:26:29.264865 140449214740288 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_50/model.json
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0328 23:26:29.265057 140449214740288 calibrate.py:118] Post calibration analysis complete.
    W0328 23:26:29.265783 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=5.9458 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.266022 140449214740288 calibrate.py:131] feature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=5.9458 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.266428 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3704 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.266630 140449214740288 calibrate.py:131] feature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3704 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.267089 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.2568 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.267289 140449214740288 calibrate.py:131] feature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.2568 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.267678 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3621 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.267869 140449214740288 calibrate.py:131] feature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3621 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.268326 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.4123 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.268521 140449214740288 calibrate.py:131] feature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.4123 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.268913 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2821 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.269103 140449214740288 calibrate.py:131] feature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2821 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.269515 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.9841 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.269702 140449214740288 calibrate.py:131] feature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.9841 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.270093 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2734 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.270280 140449214740288 calibrate.py:131] feature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2734 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.270668 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.7013 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.270851 140449214740288 calibrate.py:131] feature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.7013 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.271238 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2519 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.271429 140449214740288 calibrate.py:131] feature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2519 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.271625 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mclassifier.0._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=11.6545 calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.271768 140449214740288 calibrate.py:131] classifier.0._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=11.6545 calibrator=MaxCalibrator scale=1.0 quant)
    W0328 23:26:29.271936 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([1024, 1]).
    [32mINFO    [0m [34mclassifier.0._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.0158, 0.4703](1024) calibrator=MaxCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.272221 140449214740288 calibrate.py:131] classifier.0._weight_quantizer          : TensorQuantizer(8bit fake axis=0 amax=[0.0158, 0.4703](1024) calibrator=MaxCalibrator scale=1.0 quant)
    W0328 23:26:29.272616 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mclassifier.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=38.3167 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.272804 140449214740288 calibrate.py:131] classifier.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=38.3167 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.273202 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mclassifier.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.1175 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.273392 140449214740288 calibrate.py:131] classifier.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.1175 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.273818 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mlast_layer._input_quantizer             : TensorQuantizer(8bit fake per-tensor amax=19.2420 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.274000 140449214740288 calibrate.py:131] last_layer._input_quantizer             : TensorQuantizer(8bit fake per-tensor amax=19.2420 calibrator=HistogramCalibrator scale=1.0 quant)
    W0328 23:26:29.274388 140449214740288 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mlast_layer._weight_quantizer            : TensorQuantizer(8bit fake per-tensor amax=0.1626 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0328 23:26:29.274579 140449214740288 calibrate.py:131] last_layer._weight_quantizer            : TensorQuantizer(8bit fake per-tensor amax=0.1626 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.9...[0m
    I0328 23:26:29.275199 140449214740288 calibrate.py:105] Performing post calibration analysis for calibrator percentile_99.9...
    [32mINFO    [0m [34mStarting transformation analysis on vgg7[0m
    I0328 23:26:29.275390 140449214740288 runtime_analysis.py:357] Starting transformation analysis on vgg7
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.92097   |
    |      Average Precision       |   0.91959   |
    |        Average Recall        |   0.91991   |
    |       Average F1 Score       |   0.91957   |
    |         Average Loss         |   0.23991   |
    |       Average Latency        |  18.132 ms  |
    |   Average GPU Power Usage    |  57.867 W   |
    | Inference Energy Consumption | 0.29145 mWh |
    +------------------------------+-------------+[0m
    I0328 23:26:40.146152 140449214740288 runtime_analysis.py:521] 
    Results vgg7:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.92097   |
    |      Average Precision       |   0.91959   |
    |        Average Recall        |   0.91991   |
    |       Average F1 Score       |   0.91957   |
    |         Average Loss         |   0.23991   |
    |       Average Latency        |  18.132 ms  |
    |   Average GPU Power Usage    |  57.867 W   |
    | Inference Energy Consumption | 0.29145 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_51/model.json[0m
    I0328 23:26:40.148627 140449214740288 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_51/model.json
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0328 23:26:40.148960 140449214740288 calibrate.py:118] Post calibration analysis complete.
    [32mINFO    [0m [34mSucceeded in calibrating the model in PyTorch![0m
    I0328 23:26:40.149318 140449214740288 calibrate.py:213] Succeeded in calibrating the model in PyTorch!
    [33mWARNING [0m [34mFine tuning is disabled in the config. Skipping QAT fine tuning.[0m
    W0328 23:26:40.155524 140449214740288 fine_tune.py:92] Fine tuning is disabled in the config. Skipping QAT fine tuning.
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0328 23:26:40.159881 140449214740288 quantize.py:209] Converting PyTorch model to ONNX...
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:363: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax < 0:
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:366: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      max_bound = torch.tensor((2.0**(num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:376: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
    /root/anaconda3/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:382: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax <= epsilon:
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_43/model.onnx[0m
    I0328 23:26:48.207860 140449214740288 quantize.py:239] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_43/model.onnx
    [32mINFO    [0m [34mConverting PyTorch model to TensorRT...[0m
    I0328 23:26:48.210374 140449214740288 quantize.py:102] Converting PyTorch model to TensorRT...
    [03/28/2024-23:26:55] [TRT] [W] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
    [32mINFO    [0m [34mTensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_40/model.trt[0m
    I0328 23:06:32.223787 139939454809920 quantize.py:202] TensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_40/model.trt
    [32mINFO    [0m [34mTensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_41/model.json[0m
    I0328 23:06:32.581682 139939454809920 quantize.py:259] TensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/2024-03-28/version_41/model.json
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_46/model.json[0m
    I0328 23:06:47.114224 139939454809920 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/mase_graph/version_46/model.json
    [32mINFO    [0m [34mStarting transformation analysis on vgg7-trt_quantized[0m
    I0328 23:06:47.208054 139939454809920 runtime_analysis.py:357] Starting transformation analysis on vgg7-trt_quantized
    [32mINFO    [0m [34m
    Results vgg7-trt_quantized:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.91823   |
    |      Average Precision       |   0.9121    |
    |        Average Recall        |   0.90467   |
    |       Average F1 Score       |   0.92419   |
    |         Average Loss         |   0.24202   |
    |       Average Latency        |  8.2307 ms  |
    |   Average GPU Power Usage    |  55.687 W   |
    | Inference Energy Consumption | 0.12102 mWh |
    +------------------------------+-------------+[0m
    I0328 23:07:00.676242 139939454809920 runtime_analysis.py:521] 
    Results vgg7-trt_quantized:
    +------------------------------+-------------+
    |      Metric (Per Batch)      |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.91823   |
    |      Average Precision       |   0.9121    |
    |        Average Recall        |   0.90467   |
    |       Average F1 Score       |   0.92419   |
    |         Average Loss         |   0.24202   |
    |       Average Latency        |  8.2307 ms  |
    |   Average GPU Power Usage    |  55.687 W   |
    | Inference Energy Consumption |  0.12102 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mRuntime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/tensorrt/version_8/model.json[0m
    I0328 23:07:00.677799 139939454809920 runtime_analysis.py:143] Runtime analysis results saved to /root/mase_output/tensorrt/quantization/vgg7_cls_cifar10_2024-03-28/tensorrt/version_8/model.json


In this case, we can see through the quantized summary that one convolutional layer (feature_layers_1) has not been quantized as its precision will be configured to 'fp16' in the tensorrt engine conversion stage whilst the remaining convolutional and linear layers have been quantized.
