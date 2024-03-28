# Welcome To the TensorRT Quantization Tutorial!

This notebook is designed to show the features of the TensorRT passes integrated into MASE as part of the MASERT framework.

## Section 1. int8 Quantization
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

    /root/anaconda3/envs/mase/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


    [2024-03-27 23:37:25,708] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)


    [32mINFO    [0m [34mSet logging level to info[0m
    WARNING: Logging before flag parsing goes to stderr.
    I0327 23:37:27.921741 139859781453632 logger.py:44] Set logging level to info


Next, we load in the toml file used for quantization. To view the configuration, click [here](../../machop/configs/tensorrt/jsc_toy_int8_quantization_by_type.toml), or read the documentation on Mase [here]().


```python
# Path to your TOML file
toml_file_path = '../../../machop/configs/tensorrt/jsc_toy_int8_quantization_by_type.toml'

# Reading TOML file and converting it into a Python dictionary
with open(toml_file_path, 'r') as toml_file:
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
!ch train --config ../../../machop/configs/tensorrt/jsc_toy_int8_quantization_by_type.toml
```

    [2024-03-18 11:44:36,191] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0318 11:44:37.795595 139634499045184 seed.py:54] Seed set to 0
    +-------------------------+------------------------+--------------+-----------------+------------------------+
    | Name                    |        Default         | Config. File | Manual Override |       Effective        |
    +-------------------------+------------------------+--------------+-----------------+------------------------+
    | task                    |     [38;5;8mclassification[0m     |     cls      |                 |          cls           |
    | load_name               |          None          |              |                 |          None          |
    | load_type               |           mz           |              |                 |           mz           |
    | batch_size              |          [38;5;8m128[0m           |     256      |                 |          256           |
    | to_debug                |         False          |              |                 |         False          |
    | log_level               |          info          |              |                 |          info          |
    | report_to               |      tensorboard       |              |                 |      tensorboard       |
    | seed                    |           0            |              |                 |           0            |
    | quant_config            |          None          |              |                 |          None          |
    | training_optimizer      |          adam          |              |                 |          adam          |
    | trainer_precision       |        16-mixed        |              |                 |        16-mixed        |
    | learning_rate           |         [38;5;8m1e-05[0m          |    0.001     |                 |         0.001          |
    | weight_decay            |           0            |              |                 |           0            |
    | max_epochs              |           [38;5;8m20[0m           |      10      |                 |           10           |
    | max_steps               |           -1           |              |                 |           -1           |
    | accumulate_grad_batches |           1            |              |                 |           1            |
    | log_every_n_steps       |           50           |              |                 |           50           |
    | num_workers             |           32           |              |                 |           32           |
    | num_devices             |           1            |              |                 |           1            |
    | num_nodes               |           1            |              |                 |           1            |
    | accelerator             |          [38;5;8mauto[0m          |     gpu      |                 |          gpu           |
    | strategy                |          auto          |              |                 |          auto          |
    | is_to_auto_requeue      |         False          |              |                 |         False          |
    | github_ci               |         False          |              |                 |         False          |
    | disable_dataset_cache   |         False          |              |                 |         False          |
    | target                  |  xcu250-figd2104-2L-e  |              |                 |  xcu250-figd2104-2L-e  |
    | num_targets             |          100           |              |                 |          100           |
    | is_pretrained           |         False          |              |                 |         False          |
    | max_token_len           |          512           |              |                 |          512           |
    | project_dir             | /root/mase/mase_output |              |                 | /root/mase/mase_output |
    | project                 |          None          |              |                 |          None          |
    | model                   |          [38;5;8mNone[0m          |   jsc-toy    |                 |        jsc-toy         |
    | dataset                 |          [38;5;8mNone[0m          |     jsc      |                 |          jsc           |
    | t_max                   |           20           |              |                 |           20           |
    | eta_min                 |         1e-06          |              |                 |         1e-06          |
    +-------------------------+------------------------+--------------+-----------------+------------------------+
    [32mINFO    [0m [34mInitialising model 'jsc-toy'...[0m
    I0318 11:44:37.804295 139634499045184 cli.py:841] Initialising model 'jsc-toy'...
    [32mINFO    [0m [34mInitialising dataset 'jsc'...[0m
    I0318 11:44:37.828583 139634499045184 cli.py:869] Initialising dataset 'jsc'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-18[0m
    I0318 11:44:37.829264 139634499045184 cli.py:905] Project will be created at /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-18
    [32mINFO    [0m [34mTraining model 'jsc-toy'...[0m
    I0318 11:44:37.862516 139634499045184 cli.py:276] Training model 'jsc-toy'...
    [32mINFO    [0m [34m##### WEIGHT DECAY ##### 0[0m
    I0318 11:44:37.863071 139634499045184 cli.py:320] ##### WEIGHT DECAY ##### 0
    INFO: Using 16bit Automatic Mixed Precision (AMP)
    I0318 11:44:37.943161 139634499045184 rank_zero.py:64] Using 16bit Automatic Mixed Precision (AMP)
    INFO: GPU available: True (cuda), used: True
    I0318 11:44:37.951574 139634499045184 rank_zero.py:64] GPU available: True (cuda), used: True
    INFO: TPU available: False, using: 0 TPU cores
    I0318 11:44:37.965183 139634499045184 rank_zero.py:64] TPU available: False, using: 0 TPU cores
    INFO: IPU available: False, using: 0 IPUs
    I0318 11:44:37.965238 139634499045184 rank_zero.py:64] IPU available: False, using: 0 IPUs
    INFO: HPU available: False, using: 0 HPUs
    I0318 11:44:37.965278 139634499045184 rank_zero.py:64] HPU available: False, using: 0 HPUs
    I0318 11:44:40.324936 139634499045184 cuda.py:61] LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    I0318 11:44:40.413176 139634499045184 model_summary.py:94] 
      | Name      | Type               | Params
    -------------------------------------------------
    0 | model     | JSC_Toy            | 327   
    1 | loss_fn   | CrossEntropyLoss   | 0     
    2 | acc_train | MulticlassAccuracy | 0     
    3 | loss_val  | MeanMetric         | 0     
    4 | loss_test | MeanMetric         | 0     
    -------------------------------------------------
    327       Trainable params
    0         Non-trainable params
    327       Total params
    0.001     Total estimated model params size (MB)
    Epoch 0: 100%|█| 3084/3084 [00:21<00:00, 140.80it/s, v_num=0, train_acc_step=0.7
    Validation: |                                             | 0/? [00:00<?, ?it/s][A
    Validation:   0%|                                      | 0/3084 [00:00<?, ?it/s][A
    Validation DataLoader 0:   0%|                         | 0/3084 [00:00<?, ?it/s][A
    Validation DataLoader 0:   0%|                 | 1/3084 [00:00<04:57, 10.37it/s][A

Then we load in the checkpoint. You will have to adjust this according to where it has been stored in the mase_output directory.


```python
# Load in the trained checkpoint - change this accordingly
JSC_CHECKPOINT_PATH = "../../../mase_output/jsc-toy-cls_jsc/best.ckpt"

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

    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from ../../../mase_output/jsc-toy-cls_jsc/best.ckpt[0m
    I0318 13:22:01.808165 139755614467904 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from ../../../mase_output/jsc-toy-cls_jsc/best.ckpt


### Section 1.1 Fake Quantization

Firstly, we fake quantize the module in order to perform calibration and fine tuning before actually quantizing - this is only used if we have int8 calibration as other precisions are not currently supported within [pytorch-quantization](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#) library.

This is acheived through the `tensorrt_fake_quantize_transform_pass` which goes through the model, either by type or by name, replaces each layer appropriately to a fake quantized form if the `quantize` parameter is set in the default config (`passes.tensorrt_quantize.default.config`) or on a per name or type basis. 

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
    I0318 13:22:13.968679 139755614467904 utils.py:167] Applying fake quantization to PyTorch model...
    [32mINFO    [0m [34mFake quantization applied to PyTorch model.[0m
    I0318 13:22:14.145424 139755614467904 utils.py:192] Fake quantization applied to PyTorch model.
    [32mINFO    [0m [34mQuantized graph histogram:[0m
    I0318 13:22:14.156768 139755614467904 summary.py:84] Quantized graph histogram:
    [32mINFO    [0m [34m
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         3 |           0 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |[0m
    I0318 13:22:14.158703 139755614467904 summary.py:85] 
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         3 |           0 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |


As you can see we have succesfully quantized all linear layers inside `jsc-toy`. See [Section 4](#section-4-layer-wise-mixed-precision) for how to apply quantization layerwise.

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
    I0318 13:22:19.642365 139755614467904 calibrate.py:91] Starting calibration of the model in PyTorch...
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 13:22:19.651999 139755614467904 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 13:22:19.653707 139755614467904 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 13:22:19.655657 139755614467904 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 13:22:19.657295 139755614467904 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 13:22:19.658708 139755614467904 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 13:22:19.661043 139755614467904 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 13:22:19.846983 139755614467904 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 13:22:19.848139 139755614467904 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 13:22:19.848706 139755614467904 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 13:22:19.849354 139755614467904 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 13:22:19.849881 139755614467904 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 13:22:19.850522 139755614467904 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 13:22:19.851068 139755614467904 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 13:22:19.851693 139755614467904 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 13:22:19.852017 139755614467904 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 13:22:19.852626 139755614467904 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 13:22:19.852959 139755614467904 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 13:22:19.854428 139755614467904 tensor_quantizer.py:174] Disable HistogramCalibrator
    W0318 13:22:19.862320 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    W0318 13:22:19.862721 139755614467904 tensor_quantizer.py:239] Call .cuda() if running on GPU after loading calibrated amax.
    [32mINFO    [0m [34mseq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.9824 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:19.863394 139755614467904 calibrate.py:79] seq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.9824 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:19.864465 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.7247 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:19.864872 139755614467904 calibrate.py:79] seq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.7247 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:19.866095 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.3918 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:19.866499 139755614467904 calibrate.py:79] seq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.3918 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:19.867702 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5201 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:19.868119 139755614467904 calibrate.py:79] seq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5201 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:19.869469 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=1.7210 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:19.869887 139755614467904 calibrate.py:79] seq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=1.7210 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:19.871118 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5546 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:19.871570 139755614467904 calibrate.py:79] seq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5546 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.0...[0m
    I0318 13:22:19.872970 139755614467904 calibrate.py:53] Performing post calibration analysis for calibrator percentile_99.0...
    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 13:22:19.873783 139755614467904 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+-------------+
    |            Metric            |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.72115   |
    |      Average Precision       |   0.73523   |
    |        Average Recall        |   0.71953   |
    |       Average F1 Score       |   0.72267   |
    |         Average Loss         |   0.80283   |
    |       Average Latency        |  1.9909 ms  |
    |   Average GPU Power Usage    |  54.084 W   |
    | Inference Energy Consumption | 0.02991 mWh |
    +------------------------------+-------------+[0m
    I0318 13:22:23.013608 139755614467904 analysis.py:330] 
    Results jsc-toy:
    +------------------------------+-------------+
    |            Metric            |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.72115   |
    |      Average Precision       |   0.73523   |
    |        Average Recall        |   0.71953   |
    |       Average F1 Score       |   0.72267   |
    |         Average Loss         |   0.80283   |
    |       Average Latency        |  1.9909 ms  |
    |   Average GPU Power Usage    |  54.084 W   |
    | Inference Energy Consumption | 0.02991 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0318 13:22:23.016494 139755614467904 calibrate.py:66] Post calibration analysis complete.
    W0318 13:22:23.018086 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=4.1848 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:23.018626 139755614467904 calibrate.py:79] seq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=4.1848 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:23.020167 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.7462 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:23.020888 139755614467904 calibrate.py:79] seq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.7462 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:23.022411 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=3.8256 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:23.023213 139755614467904 calibrate.py:79] seq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=3.8256 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:23.024677 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5201 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:23.025231 139755614467904 calibrate.py:79] seq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5201 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:23.026346 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.2771 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:23.026892 139755614467904 calibrate.py:79] seq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.2771 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:23.027963 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5546 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:23.028511 139755614467904 calibrate.py:79] seq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5546 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.9...[0m
    I0318 13:22:23.029680 139755614467904 calibrate.py:53] Performing post calibration analysis for calibrator percentile_99.9...
    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 13:22:23.030415 139755614467904 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+-------------+
    |            Metric            |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.73235   |
    |      Average Precision       |   0.74538   |
    |        Average Recall        |   0.73063   |
    |       Average F1 Score       |   0.73399   |
    |         Average Loss         |   0.76725   |
    |       Average Latency        |  1.9984 ms  |
    |   Average GPU Power Usage    |  57.412 W   |
    | Inference Energy Consumption | 0.03187 mWh |
    +------------------------------+-------------+[0m
    I0318 13:22:25.768678 139755614467904 analysis.py:330] 
    Results jsc-toy:
    +------------------------------+-------------+
    |            Metric            |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.73235   |
    |      Average Precision       |   0.74538   |
    |        Average Recall        |   0.73063   |
    |       Average F1 Score       |   0.73399   |
    |         Average Loss         |   0.76725   |
    |       Average Latency        |  1.9984 ms  |
    |   Average GPU Power Usage    |  57.412 W   |
    | Inference Energy Consumption | 0.03187 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0318 13:22:25.770843 139755614467904 calibrate.py:66] Post calibration analysis complete.
    W0318 13:22:25.771903 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.7926 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:25.772400 139755614467904 calibrate.py:79] seq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.7926 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:25.773333 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.7462 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:25.773803 139755614467904 calibrate.py:79] seq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.7462 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:25.774759 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.7153 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:25.775239 139755614467904 calibrate.py:79] seq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.7153 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:25.776401 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5201 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:25.777046 139755614467904 calibrate.py:79] seq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5201 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:25.778368 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.8661 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:25.779024 139755614467904 calibrate.py:79] seq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.8661 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:25.780269 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5546 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:25.780915 139755614467904 calibrate.py:79] seq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5546 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.99...[0m
    I0318 13:22:25.782278 139755614467904 calibrate.py:53] Performing post calibration analysis for calibrator percentile_99.99...
    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 13:22:25.783110 139755614467904 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+-------------+
    |            Metric            |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.73267   |
    |      Average Precision       |   0.74556   |
    |        Average Recall        |   0.73097   |
    |       Average F1 Score       |   0.73429   |
    |         Average Loss         |   0.76364   |
    |       Average Latency        |  2.0014 ms  |
    |   Average GPU Power Usage    |  57.504 W   |
    | Inference Energy Consumption | 0.03197 mWh |
    +------------------------------+-------------+[0m
    I0318 13:22:28.523978 139755614467904 analysis.py:330] 
    Results jsc-toy:
    +------------------------------+-------------+
    |            Metric            |    Value    |
    +------------------------------+-------------+
    | Average Validation Accuracy  |   0.73267   |
    |      Average Precision       |   0.74556   |
    |        Average Recall        |   0.73097   |
    |       Average F1 Score       |   0.73429   |
    |         Average Loss         |   0.76364   |
    |       Average Latency        |  2.0014 ms  |
    |   Average GPU Power Usage    |  57.504 W   |
    | Inference Energy Consumption | 0.03197 mWh |
    +------------------------------+-------------+
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0318 13:22:28.526261 139755614467904 calibrate.py:66] Post calibration analysis complete.
    W0318 13:22:29.754698 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=7.1553 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:29.755807 139755614467904 calibrate.py:79] seq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=7.1553 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:30.601948 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.7464 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:30.602879 139755614467904 calibrate.py:79] seq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.7464 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:31.914243 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=6.4895 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:31.915047 139755614467904 calibrate.py:79] seq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=6.4895 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:32.743558 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5179 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:32.744498 139755614467904 calibrate.py:79] seq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5179 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:34.152468 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=3.3733 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:34.153729 139755614467904 calibrate.py:79] seq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=3.3733 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:35.017431 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5531 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:35.018441 139755614467904 calibrate.py:79] seq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5531 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator mse...[0m
    I0318 13:22:35.020324 139755614467904 calibrate.py:53] Performing post calibration analysis for calibrator mse...
    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 13:22:35.021559 139755614467904 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+--------------+
    |            Metric            |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.73272    |
    |      Average Precision       |   0.74538    |
    |        Average Recall        |   0.73103    |
    |       Average F1 Score       |    0.7343    |
    |         Average Loss         |   0.76502    |
    |       Average Latency        |  1.9966 ms   |
    |   Average GPU Power Usage    |   56.178 W   |
    | Inference Energy Consumption | 0.031157 mWh |
    +------------------------------+--------------+[0m
    I0318 13:22:38.135719 139755614467904 analysis.py:330] 
    Results jsc-toy:
    +------------------------------+--------------+
    |            Metric            |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.73272    |
    |      Average Precision       |   0.74538    |
    |        Average Recall        |   0.73103    |
    |       Average F1 Score       |    0.7343    |
    |         Average Loss         |   0.76502    |
    |       Average Latency        |  1.9966 ms   |
    |   Average GPU Power Usage    |   56.178 W   |
    | Inference Energy Consumption | 0.031157 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0318 13:22:38.138723 139755614467904 calibrate.py:66] Post calibration analysis complete.
    W0318 13:22:43.433615 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=6.2402 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:43.434860 139755614467904 calibrate.py:79] seq_blocks.2._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=6.2402 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:45.434811 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.7465 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:45.436088 139755614467904 calibrate.py:79] seq_blocks.2._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.7465 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:51.187860 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.7175 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:51.189001 139755614467904 calibrate.py:79] seq_blocks.5._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=5.7175 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:53.107534 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5203 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:53.108866 139755614467904 calibrate.py:79] seq_blocks.5._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5203 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:22:59.775393 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.8672 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:22:59.776729 139755614467904 calibrate.py:79] seq_blocks.8._input_quantizer           : TensorQuantizer(8bit fake per-tensor amax=2.8672 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 13:23:01.662585 139755614467904 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mseq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5548 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 13:23:01.663614 139755614467904 calibrate.py:79] seq_blocks.8._weight_quantizer          : TensorQuantizer(8bit fake per-tensor amax=0.5548 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator entropy...[0m
    I0318 13:23:01.665429 139755614467904 calibrate.py:53] Performing post calibration analysis for calibrator entropy...
    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 13:23:01.666694 139755614467904 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+--------------+
    |            Metric            |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.73316    |
    |      Average Precision       |    0.746     |
    |        Average Recall        |   0.73142    |
    |       Average F1 Score       |   0.73475    |
    |         Average Loss         |   0.76419    |
    |       Average Latency        |  2.0012 ms   |
    |   Average GPU Power Usage    |   56.568 W   |
    | Inference Energy Consumption | 0.031446 mWh |
    +------------------------------+--------------+[0m
    I0318 13:23:04.402893 139755614467904 analysis.py:330] 
    Results jsc-toy:
    +------------------------------+--------------+
    |            Metric            |    Value     |
    +------------------------------+--------------+
    | Average Validation Accuracy  |   0.73316    |
    |      Average Precision       |    0.746     |
    |        Average Recall        |   0.73142    |
    |       Average F1 Score       |   0.73475    |
    |         Average Loss         |   0.76419    |
    |       Average Latency        |  2.0012 ms   |
    |   Average GPU Power Usage    |   56.568 W   |
    | Inference Energy Consumption | 0.031446 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0318 13:23:04.405836 139755614467904 calibrate.py:66] Post calibration analysis complete.
    [32mINFO    [0m [34mSucceeded in calibrating the model in PyTorch![0m
    I0318 13:23:04.407040 139755614467904 calibrate.py:159] Succeeded in calibrating the model in PyTorch!


From the results, the 99% `percentile` clips too many values during the amax calibration, compromising the loss. However 99.99% demonstrates higher validation accuracy alongside `mse` and `entropy` for `jsc-toy`. For such a small model, the methods are not highly distinguished, however for larger models this calibration process will be important for ensuring the quantized model still performs well. 

### Section 1.3 Quantized Aware Training (QAT)

The `tensorrt_fine_tune_transform_pass` is used to fine tune the quantized model. 

For QAT it is typical to employ 10% of the original training epochs, starting at 1% of the initial training learning rate, and a cosine annealing learning rate schedule that follows the decreasing half of a cosine period, down to 1% of the initial fine tuning learning rate (0.01% of the initial training learning rate). However this default can be overidden by setting the `epochs`, `initial_learning_rate` and `final_learning_rate` in `passes.tensorrt_quantize.fine_tune`.

The fine tuned checkpoints are stored in the ckpts/fine_tuning folder:

```
mase_output
└── tensorrt
    └── model_task_dataset_date
        └──quantization
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

    [32mINFO    [0m [34mStarting Fine Tuning...[0m
    I0318 13:23:20.128491 139755614467904 fine_tune.py:62] Starting Fine Tuning...
    [32mINFO    [0m [34mFine tuninig for 2 epochs[0m
    I0318 13:23:20.170033 139755614467904 fine_tune.py:102] Fine tuninig for 2 epochs
    I0318 13:23:20.252301 139755614467904 rank_zero.py:64] GPU available: True (cuda), used: True
    I0318 13:23:20.266848 139755614467904 rank_zero.py:64] TPU available: False, using: 0 TPU cores
    I0318 13:23:20.267485 139755614467904 rank_zero.py:64] IPU available: False, using: 0 IPUs
    I0318 13:23:20.268031 139755614467904 rank_zero.py:64] HPU available: False, using: 0 HPUs
    I0318 13:23:20.274132 139755614467904 rank_zero.py:64] You are using a CUDA device ('NVIDIA GeForce RTX 3070') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    I0318 13:23:22.531366 139755614467904 cuda.py:61] LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    I0318 13:23:22.546064 139755614467904 model_summary.py:94] 
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


                                                                               

    /opt/conda/envs/mase/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:441: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=31` in the `DataLoader` to improve performance.
    /opt/conda/envs/mase/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=31` in the `DataLoader` to improve performance.


    Epoch 1: 100%|██████████| 3084/3084 [00:57<00:00, 53.90it/s, v_num=4, train_acc_step=0.765, val_acc_epoch=0.733, val_loss_epoch=0.750]

    I0318 13:25:17.216056 139755614467904 rank_zero.py:64] `Trainer.fit` stopped: `max_epochs=2` reached.


    Epoch 1: 100%|██████████| 3084/3084 [00:57<00:00, 53.90it/s, v_num=4, train_acc_step=0.765, val_acc_epoch=0.733, val_loss_epoch=0.750]

    [32mINFO    [0m [34mFine Tuning Complete[0m
    I0318 13:25:17.223005 139755614467904 fine_tune.py:121] Fine Tuning Complete


    


### Section 1.4 TensorRT Quantization

After QAT, we are now ready to convert the model to a tensorRT engine so that it can be run with the superior inference speeds. To do so, we use the `tensorrt_engine_interface_pass` which converts the `MaseGraph`'s model from a Pytorch one to an ONNX format as an intermediate stage of the conversion.

During the conversion process, the `.onnx` and `.trt` files are stored to their respective folders shown in [Section 1.3](#section-13-quantized-aware-training-qat).

This interface pass returns a dictionary containing the `onnx_path` and `trt_engine_path`.


```python
mg, meta = tensorrt_engine_interface_pass(mg, pass_args=tensorrt_config)
```

    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0318 13:27:06.409828 139755614467904 quantize.py:129] Converting PyTorch model to ONNX...
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/onnx/2024_03_18/version_10/model.onnx[0m
    I0318 13:27:31.306156 139755614467904 quantize.py:152] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/onnx/2024_03_18/version_10/model.onnx
    [32mINFO    [0m [34mConverting PyTorch model to TensorRT...[0m
    I0318 13:27:37.806547 139755614467904 quantize.py:55] Converting PyTorch model to TensorRT...
    [32mINFO    [0m [34mTensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/trt/2024_03_18/version_6/model.trt[0m
    I0318 13:27:46.186066 139755614467904 quantize.py:124] TensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/trt/2024_03_18/version_6/model.trt
    [32mINFO    [0m [34mTensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/json/2024_03_18/version_6/model.json[0m
    I0318 13:27:53.259016 139755614467904 quantize.py:168] TensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/json/2024_03_18/version_6/model.json


### Section 1.5 Performance Analysis

To showcase the improved inference speeds and to evaluate accuracy and other performance metrics, the `tensorrt_analysis_pass` can be used.

The tensorRT engine path obtained the previous interface pass is now inputted into the the analysis pass. The same pass can take a MaseGraph as an input, as well as an ONNX graph. For this comparison, we will first run the anaylsis pass on the original unquantized model and then on the int8 quantized model.


```python
_, _ = runtime_analysis_pass(mg_original, pass_args=runtime_analysis_config)
_, _ = runtime_analysis_pass(meta['trt_engine_path'], pass_args=runtime_analysis_config)
```

    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 13:28:06.032049 139755614467904 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results jsc-toy:
    +------------------------------+--------------+
    |            Metric            |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.73168    |
    |      Average Precision       |   0.74472    |
    |        Average Recall        |   0.73037    |
    |       Average F1 Score       |   0.73369    |
    |         Average Loss         |   0.76315    |
    |       Average Latency        |  1.0634 ms   |
    |   Average GPU Power Usage    |   59.251 W   |
    | Inference Energy Consumption | 0.017502 mWh |
    +------------------------------+--------------+[0m
    I0318 13:28:11.978660 139755614467904 analysis.py:330] 
    Results jsc-toy:
    +------------------------------+--------------+
    |            Metric            |    Value     |
    +------------------------------+--------------+
    |    Average Test Accuracy     |   0.73168    |
    |      Average Precision       |   0.74472    |
    |        Average Recall        |   0.73037    |
    |       Average F1 Score       |   0.73369    |
    |         Average Loss         |   0.76315    |
    |       Average Latency        |  1.0634 ms   |
    |   Average GPU Power Usage    |   59.251 W   |
    | Inference Energy Consumption | 0.017502 mWh |
    +------------------------------+--------------+
    [32mINFO    [0m [34m
    TensorRT Engine Input/Output Information:
    Index | Type    | DataType | Static Shape         | Dynamic Shape        | Name
    ------|---------|----------|----------------------|----------------------|-----------------------
    0     | Input   | FLOAT    | (256, 16)              | (256, 16)              | input
    1     | Output  | FLOAT    | (256, 5)               | (256, 5)               | 109[0m
    I0318 13:28:11.995075 139755614467904 analysis.py:117] 
    TensorRT Engine Input/Output Information:
    Index | Type    | DataType | Static Shape         | Dynamic Shape        | Name
    ------|---------|----------|----------------------|----------------------|-----------------------
    0     | Input   | FLOAT    | (256, 16)              | (256, 16)              | input
    1     | Output  | FLOAT    | (256, 5)               | (256, 5)               | 109
    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 13:28:11.996511 139755614467904 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results jsc-toy-quantized:
    +------------------------------+---------------+
    |            Metric            |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.73394    |
    |      Average Precision       |    0.74894    |
    |        Average Recall        |    0.73414    |
    |       Average F1 Score       |    0.73757    |
    |         Average Loss         |    0.75233    |
    |       Average Latency        |  0.20819 ms   |
    |   Average GPU Power Usage    |   59.366 W    |
    | Inference Energy Consumption | 0.0034331 mWh |
    +------------------------------+---------------+[0m
    I0318 13:28:17.535370 139755614467904 analysis.py:330] 
    Results jsc-toy-quantized:
    +------------------------------+---------------+
    |            Metric            |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.73394    |
    |      Average Precision       |    0.74894    |
    |        Average Recall        |    0.73414    |
    |       Average F1 Score       |    0.73757    |
    |         Average Loss         |    0.75233    |
    |       Average Latency        |  0.20819 ms   |
    |   Average GPU Power Usage    |   59.366 W    |
    | Inference Energy Consumption | 0.0034331 mWh |
    +------------------------------+---------------+


As shown above, the latency has decreased around 4x with the `jsc-toy` model without compromising accuracy due to the well calibrated amax and quantization-aware fine tuning. The inference energy consumption has thus also dropped tremendously and this is an excellent demonstration for the need to quantize in industry especially for LLMs in order to reduce energy usage. 

## Section 2. fp16 Quantization

We will now load in a new toml configuration that uses fp16 instead of int8, whilst keeping the other settings the exact same for a fair comparison. This time however, we will use chop from the terminal which runs all the passes showcased in [Section 1](#section-1---int8-quantization).

Since float quantization does not require calibration, nor is it supported by `pytorch-quantization`, the model will not undergo fake quantization; for the time being this unfortunately means QAT is unavailable and only undergoes Post Training Quantization (PTQ). 


```python
!ch transform --config ../../../machop/configs/tensorrt/jsc_toy_fp16_quantization_by_type.toml --load {JSC_CHECKPOINT_PATH} --load-type pl
```

    567.49s - pydevd: Sending message related to process being replaced timed-out after 5 seconds
    [2024-03-18 13:29:23,372] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0318 13:29:24.963454 139971211573056 seed.py:54] Seed set to 0
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | Name                    |        Default         | Config. File |     Manual Override      |        Effective         |
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | task                    |     [38;5;8mclassification[0m     |     cls      |                          |           cls            |
    | load_name               |          [38;5;8mNone[0m          |              | /root/mase/mase_output/j | /root/mase/mase_output/j |
    |                         |                        |              | sc-toy-cls_jsc/best.ckpt | sc-toy-cls_jsc/best.ckpt |
    | load_type               |           [38;5;8mmz[0m           |              |            pl            |            pl            |
    | batch_size              |          [38;5;8m128[0m           |     256      |                          |           256            |
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
    | num_workers             |           32           |              |                          |            32            |
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
    I0318 13:29:24.973379 139971211573056 cli.py:841] Initialising model 'jsc-toy'...
    [32mINFO    [0m [34mInitialising dataset 'jsc'...[0m
    I0318 13:29:24.996573 139971211573056 cli.py:869] Initialising dataset 'jsc'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-18[0m
    I0318 13:29:24.997021 139971211573056 cli.py:905] Project will be created at /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-18
    [32mINFO    [0m [34mTransforming model 'jsc-toy'...[0m
    I0318 13:29:25.030123 139971211573056 cli.py:365] Transforming model 'jsc-toy'...
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/jsc-toy-cls_jsc/best.ckpt[0m
    I0318 13:29:27.502959 139971211573056 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/jsc-toy-cls_jsc/best.ckpt
    [32mINFO    [0m [34mApplying fake quantization to PyTorch model...[0m
    I0318 13:29:30.147488 139971211573056 utils.py:167] Applying fake quantization to PyTorch model...
    [33mWARNING [0m [34mint8 precision not found in config. Skipping fake quantization.[0m
    W0318 13:29:30.147828 139971211573056 utils.py:170] int8 precision not found in config. Skipping fake quantization.
    [33mWARNING [0m [34mint8 precision not found in config. Skipping calibration.[0m
    W0318 13:29:30.147962 139971211573056 calibrate.py:86] int8 precision not found in config. Skipping calibration.
    [32mINFO    [0m [34mQuantized graph histogram:[0m
    I0318 13:29:30.171943 139971211573056 summary.py:84] Quantized graph histogram:
    [32mINFO    [0m [34m
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         0 |           3 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |[0m
    I0318 13:29:30.173280 139971211573056 summary.py:85] 
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         0 |           3 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |
    [33mWARNING [0m [34mint8 precision not found in config. Skipping QAT fine tuning.[0m
    W0318 13:29:30.174435 139971211573056 fine_tune.py:57] int8 precision not found in config. Skipping QAT fine tuning.
    [32mINFO    [0m [34mQuantized graph histogram:[0m
    I0318 13:29:30.183183 139971211573056 summary.py:84] Quantized graph histogram:
    [32mINFO    [0m [34m
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         0 |           3 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |[0m
    I0318 13:29:30.183739 139971211573056 summary.py:85] 
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         0 |           3 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0318 13:29:30.185087 139971211573056 quantize.py:129] Converting PyTorch model to ONNX...
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/onnx/2024_03_18/version_11/model.onnx[0m
    I0318 13:29:32.601623 139971211573056 quantize.py:152] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/onnx/2024_03_18/version_11/model.onnx
    [32mINFO    [0m [34mConverting PyTorch model to TensorRT...[0m
    I0318 13:29:32.601973 139971211573056 quantize.py:55] Converting PyTorch model to TensorRT...
    [32mINFO    [0m [34mTensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/trt/2024_03_18/version_7/model.trt[0m
    I0318 13:29:47.617324 139971211573056 quantize.py:124] TensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/trt/2024_03_18/version_7/model.trt
    [32mINFO    [0m [34mTensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/json/2024_03_18/version_7/model.json[0m
    I0318 13:29:47.872362 139971211573056 quantize.py:168] TensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/json/2024_03_18/version_7/model.json
    [32mINFO    [0m [34mQuantized graph histogram:[0m
    I0318 13:29:47.894693 139971211573056 summary.py:84] Quantized graph histogram:
    [32mINFO    [0m [34m
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         0 |           3 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |[0m
    I0318 13:29:47.895272 139971211573056 summary.py:85] 
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
    | Linear          | linear       |       3 |         0 |           3 |
    | ReLU            | relu         |       4 |         0 |           4 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |
    [32mINFO    [0m [34m
    TensorRT Engine Input/Output Information:
    Index | Type    | DataType | Static Shape         | Dynamic Shape        | Name
    ------|---------|----------|----------------------|----------------------|-----------------------
    0     | Input   | FLOAT    | (256, 16)              | (256, 16)              | input
    1     | Output  | FLOAT    | (256, 5)               | (256, 5)               | 37[0m
    I0318 13:29:48.377556 139971211573056 analysis.py:117] 
    TensorRT Engine Input/Output Information:
    Index | Type    | DataType | Static Shape         | Dynamic Shape        | Name
    ------|---------|----------|----------------------|----------------------|-----------------------
    0     | Input   | FLOAT    | (256, 16)              | (256, 16)              | input
    1     | Output  | FLOAT    | (256, 5)               | (256, 5)               | 37
    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 13:29:48.377788 139971211573056 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results jsc-toy-quantized:
    +------------------------------+---------------+
    |            Metric            |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.73526    |
    |      Average Precision       |    0.75069    |
    |        Average Recall        |    0.73547    |
    |       Average F1 Score       |    0.73897    |
    |         Average Loss         |    0.74842    |
    |       Average Latency        |  0.389921 ms  |
    |   Average GPU Power Usage    |    60.18 W    |
    | Inference Energy Consumption | 0.0055032 mWh |
    +------------------------------+---------------+[0m
    I0318 13:29:53.848961 139971211573056 analysis.py:330] 
    Results jsc-toy-quantized:
    +------------------------------+---------------+
    |            Metric            |     Value     |
    +------------------------------+---------------+
    |    Average Test Accuracy     |    0.73526    |
    |      Average Precision       |    0.75069    |
    |        Average Recall        |    0.73547    |
    |       Average F1 Score       |    0.73897    |
    |         Average Loss         |    0.74842    |
    |       Average Latency        |  0.389921 ms  |
    |   Average GPU Power Usage    |    60.18 W    |
    | Inference Energy Consumption | 0.0055032 mWh |
    +------------------------------+---------------+
    [32mINFO    [0m [34mSaved mase graph to /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-18/software/transform/transformed_ckpt[0m
    I0318 13:29:53.973135 139971211573056 save_and_load.py:147] Saved mase graph to /root/mase/mase_output/jsc-toy_cls_jsc_2024-03-18/software/transform/transformed_ckpt
    [32mINFO    [0m [34mTransformation is completed[0m
    I0318 13:29:53.973508 139971211573056 cli.py:383] Transformation is completed


As you can see, `fp16` acheives a slighty higher test accuracy but a slightly lower latency (~30%) from that of int8 quantization; it is still ~2.5x faster than the unquantized model. Now lets apply quantization to a more complicated model.

## Section 3. Type-wise Mixed Precision on Larger Model
We will now quantize `vgg7` which includes both convolutional and linear layers, however for this demonstration we want to quantize all layer types except the linear layers.

In this case, we set:

- The `by` parameter to `type`
- The `quantize` parameter to true for `passes.tensorrt_quantize.conv2d.config` and `precision` parameter to 'int8'.
- The `input` and `weight` quantize axis for the conv2d layers.
- The default `passes.tensorrt_quantize.default.config` precision to true. 

During the TensorRT quantization, the model's conv2d layers will be converted to an int8 fake quantized form, whilst the linear layers are kept to their default 'fp16'. Calibration of the conv2d layers will be undergone and fine tuning.  

You may either download a pretrained model [here](https://imperiallondon-my.sharepoint.com/:f:/g/personal/zz7522_ic_ac_uk/Emh3VT7Q_qRFmnp8kDrcgDoBwGUuzLwwKNtX8ZAt368jJQ?e=gsKONa), otherwise train it yourself as shown below. 


```python
!ch train --config ../../../machop/configs/tensorrt/vgg7_layerwise_mixed_precision.toml.toml
```

We will now load the checkpoint in, quantize the model and compare it to the unquantized version as we did in [Section 1.5](#section-15-performance-analysis)


```python
VGG_CHECKPOINT_PATH = "../../../mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt"
```


```python
!ch transform --config ../../../machop/configs/tensorrt/vgg7_typewise_mixed_precision.toml --load {VGG_CHECKPOINT_PATH} --load-type pl
```

    [2024-03-18 15:52:43,166] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    INFO: Seed set to 0
    WARNING: Logging before flag parsing goes to stderr.
    I0318 15:52:44.743802 139761495775040 seed.py:54] Seed set to 0
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | Name                    |        Default         | Config. File |     Manual Override      |        Effective         |
    +-------------------------+------------------------+--------------+--------------------------+--------------------------+
    | task                    |     [38;5;8mclassification[0m     |     cls      |                          |           cls            |
    | load_name               |          [38;5;8mNone[0m          |              | /root/mase/mase_output/v | /root/mase/mase_output/v |
    |                         |                        |              |  gg7-pre-trained/test-   |  gg7-pre-trained/test-   |
    |                         |                        |              |     accu-0.9332.ckpt     |     accu-0.9332.ckpt     |
    | load_type               |           [38;5;8mmz[0m           |              |            pl            |            pl            |
    | batch_size              |          [38;5;8m128[0m           |     256      |                          |           256            |
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
    | num_workers             |           32           |              |                          |            32            |
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
    I0318 15:52:44.753937 139761495775040 cli.py:841] Initialising model 'vgg7'...
    [32mINFO    [0m [34mInitialising dataset 'cifar10'...[0m
    I0318 15:52:44.878532 139761495775040 cli.py:869] Initialising dataset 'cifar10'...
    [32mINFO    [0m [34mProject will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-18[0m
    I0318 15:52:44.878995 139761495775040 cli.py:905] Project will be created at /root/mase/mase_output/vgg7_cls_cifar10_2024-03-18
    [32mINFO    [0m [34mTransforming model 'vgg7'...[0m
    I0318 15:52:44.909238 139761495775040 cli.py:365] Transforming model 'vgg7'...
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0318 15:52:50.380097 139761495775040 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from /root/mase/mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt


## Section 4. Layer-wise Mixed Precision

So far we have strictly quantized either in int8 or fp16. Now, we will show how to conduct layerwise mixed precision using the same `vgg7` model. In this case we will show how for instance, layer 0 and 1 can be set to fp16, while layer 2 and 3 can be int8 quantized. 

For this, we set:
- The `by` parameter to `name`
- The `precision` to 'fp16' for `passes.tensorrt_quantize.feature_layers_0.config and passes.tensorrt_quantize.feature_layers_1.config`
- The `precision` to 'int8' for `passes.tensorrt_quantize.feature_layers_0.config and passes.tensorrt_quantize.feature_layers_1.config`




```python
import sys
import os
from pathlib import Path
import toml
from copy import copy, deepcopy

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

    /root/anaconda3/envs/mase/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


    [2024-03-19 16:33:49,791] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)


    [32mINFO    [0m [34mSet logging level to info[0m
    WARNING: Logging before flag parsing goes to stderr.
    I0319 16:33:51.317272 140297220093760 logger.py:44] Set logging level to info



```python
# Path to your TOML file
# toml_file_path = '../../../machop/configs/tensorrt/vgg7_layerwise_mixed_precision.toml'
toml_file_path = '../../../machop/configs/tensorrt/vgg7_typewise_mixed_precision.toml'

# Reading TOML file and converting it into a Python dictionary
with open(toml_file_path, 'r') as toml_file:
    pass_args = toml.load(toml_file)

# Extract the 'passes.tensorrt' section and its children
tensorrt_config = pass_args.get('passes', {}).get('tensorrt', {})
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
configs = [tensorrt_config, runtime_analysis_config]
for config in configs:
    config['task'] = pass_args['task']
    config['batch_size'] = pass_args['batch_size']
    config['model'] = pass_args['model']
    config['data_module'] = data_module
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

    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified



```python
# Load in the trained checkpoint - change this accordingly
VGG_CHECKPOINT_PATH = "../../../mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt"

model = load_model(load_name=VGG_CHECKPOINT_PATH, load_type="pl", model=model)

# Initiate metadata
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)
mg, _ = init_metadata_analysis_pass(mg, None)

mg_original = deepcopy_mase_graph(mg)

mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)
mg, _ = metadata_value_type_cast_transform_pass(mg, pass_args={"fn": to_numpy_if_tensor})
```

    [32mINFO    [0m [34mLoaded pytorch lightning checkpoint from ../../../mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt[0m
    I0318 15:33:56.843245 140155325249344 checkpoint_load.py:85] Loaded pytorch lightning checkpoint from ../../../mase_output/vgg7-pre-trained/test-accu-0.9332.ckpt



```python
_, _ = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)
```

    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 09:55:37.107798 140195009255232 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+-------------+
    |            Metric            |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.91967   |
    |      Average Precision       |   0.92315   |
    |        Average Recall        |   0.92362   |
    |       Average F1 Score       |   0.92326   |
    |         Average Loss         |   0.23674   |
    |       Average Latency        |  28.123 ms  |
    |   Average GPU Power Usage    |  100.17 W   |
    | Inference Energy Consumption | 0.78256 mWh |
    +------------------------------+-------------+[0m
    I0318 09:55:40.969246 140195009255232 analysis.py:330] 
    Results vgg7:
    +------------------------------+-------------+
    |            Metric            |    Value    |
    +------------------------------+-------------+
    |    Average Test Accuracy     |   0.91967   |
    |      Average Precision       |   0.92315   |
    |        Average Recall        |   0.92362   |
    |       Average F1 Score       |   0.92326   |
    |         Average Loss         |   0.23674   |
    |       Average Latency        |  28.123 ms  |
    |   Average GPU Power Usage    |  100.17 W   |
    | Inference Energy Consumption | 0.78256 mWh |
    +------------------------------+-------------+



```python
mg, _ = tensorrt_fake_quantize_transform_pass(mg, pass_args=tensorrt_config)
# summarize_quantization_analysis_pass(mg_original, mg)
mg, _ = tensorrt_calibrate_transform_pass(mg, pass_args=tensorrt_config)
mg, _ = tensorrt_fine_tune_transform_pass(mg, pass_args=tensorrt_config)
mg, meta = tensorrt_engine_interface_pass(mg, pass_args=tensorrt_config)
```

    [32mINFO    [0m [34mApplying fake quantization to PyTorch model...[0m
    I0318 15:34:00.409916 140155325249344 utils.py:168] Applying fake quantization to PyTorch model...
    [32mINFO    [0m [34mFake quantization applied to PyTorch model.[0m
    I0318 15:34:00.682099 140155325249344 utils.py:193] Fake quantization applied to PyTorch model.
    [32mINFO    [0m [34mStarting calibration of the model in PyTorch...[0m
    I0318 15:34:00.683382 140155325249344 calibrate.py:91] Starting calibration of the model in PyTorch...
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.694193 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.695298 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.695932 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.696544 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.697141 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.697722 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.698303 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.698865 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.699461 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.700013 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.700582 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mDisabling Quantization and Enabling Calibration[0m
    I0318 15:34:00.701155 140155325249344 calibrate.py:100] Disabling Quantization and Enabling Calibration
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.481201 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.482733 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.483446 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.484515 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.485068 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.486183 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.486567 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.487422 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.487805 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.488553 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.488957 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.489720 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.490093 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.490822 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.491214 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.491982 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.492378 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.493196 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.493597 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.494379 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.494779 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.495552 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    [32mINFO    [0m [34mEnabling Quantization and Disabling Calibration[0m
    I0318 15:34:02.495936 140155325249344 calibrate.py:121] Enabling Quantization and Disabling Calibration
    W0318 15:34:02.496721 140155325249344 tensor_quantizer.py:174] Disable HistogramCalibrator
    W0318 15:34:02.503043 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    W0318 15:34:02.503463 140155325249344 tensor_quantizer.py:239] Call .cuda() if running on GPU after loading calibrated amax.
    [32mINFO    [0m [34mfeature_layers.0._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=2.6392 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.504164 140155325249344 calibrate.py:79] feature_layers.0._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=2.6392 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.505316 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.0._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2797 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.505767 140155325249344 calibrate.py:79] feature_layers.0._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2797 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.507040 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.3020 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.507491 140155325249344 calibrate.py:79] feature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.3020 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.508672 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2366 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.509136 140155325249344 calibrate.py:79] feature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2366 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.510375 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=1.8184 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.510823 140155325249344 calibrate.py:79] feature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=1.8184 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.512043 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2296 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.512472 140155325249344 calibrate.py:79] feature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.2296 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.513764 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.4591 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.514211 140155325249344 calibrate.py:79] feature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.4591 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.515444 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2080 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.515901 140155325249344 calibrate.py:79] feature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2080 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.517146 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.9262 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.517589 140155325249344 calibrate.py:79] feature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.9262 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.518806 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2013 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.519248 140155325249344 calibrate.py:79] feature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2013 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.520494 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.6157 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.520926 140155325249344 calibrate.py:79] feature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=1.6157 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:02.522181 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.1879 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:02.522614 140155325249344 calibrate.py:79] feature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.1879 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.0...[0m
    I0318 15:34:02.524394 140155325249344 calibrate.py:53] Performing post calibration analysis for calibrator percentile_99.0...
    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 15:34:02.525266 140155325249344 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+------------+
    |            Metric            |   Value    |
    +------------------------------+------------+
    |    Average Test Accuracy     |  0.91682   |
    |      Average Precision       |  0.91546   |
    |        Average Recall        |  0.91579   |
    |       Average F1 Score       |  0.91553   |
    |         Average Loss         |  0.25923   |
    |       Average Latency        | 36.294 ms  |
    |   Average GPU Power Usage    |  108.73 W  |
    | Inference Energy Consumption | 1.0962 mWh |
    +------------------------------+------------+[0m
    I0318 15:34:06.029327 140155325249344 analysis.py:323] 
    Results vgg7:
    +------------------------------+------------+
    |            Metric            |   Value    |
    +------------------------------+------------+
    |    Average Test Accuracy     |  0.91682   |
    |      Average Precision       |  0.91546   |
    |        Average Recall        |  0.91579   |
    |       Average F1 Score       |  0.91553   |
    |         Average Loss         |  0.25923   |
    |       Average Latency        | 36.294 ms  |
    |   Average GPU Power Usage    |  108.73 W  |
    | Inference Energy Consumption | 1.0962 mWh |
    +------------------------------+------------+
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0318 15:34:06.030981 140155325249344 calibrate.py:66] Post calibration analysis complete.
    W0318 15:34:06.032042 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.0._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=2.6392 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.032518 140155325249344 calibrate.py:79] feature_layers.0._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=2.6392 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.033420 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.0._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3434 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.033899 140155325249344 calibrate.py:79] feature_layers.0._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3434 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.034892 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=6.0151 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.035389 140155325249344 calibrate.py:79] feature_layers.3._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=6.0151 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.036347 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3704 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.036843 140155325249344 calibrate.py:79] feature_layers.3._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3704 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.037833 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.2875 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.038331 140155325249344 calibrate.py:79] feature_layers.7._input_quantizer       : TensorQuantizer(8bit fake per-tensor amax=3.2875 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.039333 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3621 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.039835 140155325249344 calibrate.py:79] feature_layers.7._weight_quantizer      : TensorQuantizer(8bit fake per-tensor amax=0.3621 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.040834 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.4120 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.041361 140155325249344 calibrate.py:79] feature_layers.10._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.4120 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.042325 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2821 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.042822 140155325249344 calibrate.py:79] feature_layers.10._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2821 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.043805 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.9829 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.044299 140155325249344 calibrate.py:79] feature_layers.14._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.9829 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.045269 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2734 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.045764 140155325249344 calibrate.py:79] feature_layers.14._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2734 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.046752 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.7024 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.047246 140155325249344 calibrate.py:79] feature_layers.17._input_quantizer      : TensorQuantizer(8bit fake per-tensor amax=2.7024 calibrator=HistogramCalibrator scale=1.0 quant)
    W0318 15:34:06.048205 140155325249344 tensor_quantizer.py:238] Load calibrated amax, shape=torch.Size([]).
    [32mINFO    [0m [34mfeature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2519 calibrator=HistogramCalibrator scale=1.0 quant)[0m
    I0318 15:34:06.048707 140155325249344 calibrate.py:79] feature_layers.17._weight_quantizer     : TensorQuantizer(8bit fake per-tensor amax=0.2519 calibrator=HistogramCalibrator scale=1.0 quant)
    [32mINFO    [0m [34mPerforming post calibration analysis for calibrator percentile_99.9...[0m
    I0318 15:34:06.050029 140155325249344 calibrate.py:53] Performing post calibration analysis for calibrator percentile_99.9...
    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 15:34:06.050704 140155325249344 analysis.py:214] Starting transformation analysis
    [32mINFO    [0m [34m
    Results vgg7:
    +------------------------------+-----------+
    |            Metric            |   Value   |
    +------------------------------+-----------+
    | Average Validation Accuracy  |  0.92441  |
    |      Average Precision       |  0.92283  |
    |        Average Recall        |  0.92325  |
    |       Average F1 Score       |  0.92293  |
    |         Average Loss         |  0.23622  |
    |       Average Latency        | 36.334 ms |
    |   Average GPU Power Usage    | 111.46 W  |
    | Inference Energy Consumption | 1.125 mWh |
    +------------------------------+-----------+[0m
    I0318 15:34:09.521739 140155325249344 analysis.py:323] 
    Results vgg7:
    +------------------------------+-----------+
    |            Metric            |   Value   |
    +------------------------------+-----------+
    | Average Validation Accuracy  |  0.92441  |
    |      Average Precision       |  0.92283  |
    |        Average Recall        |  0.92325  |
    |       Average F1 Score       |  0.92293  |
    |         Average Loss         |  0.23622  |
    |       Average Latency        | 36.334 ms |
    |   Average GPU Power Usage    | 111.46 W  |
    | Inference Energy Consumption | 1.125 mWh |
    +------------------------------+-----------+
    [32mINFO    [0m [34mPost calibration analysis complete.[0m
    I0318 15:34:09.524197 140155325249344 calibrate.py:66] Post calibration analysis complete.
    [32mINFO    [0m [34mSucceeded in calibrating the model in PyTorch![0m
    I0318 15:34:09.525355 140155325249344 calibrate.py:159] Succeeded in calibrating the model in PyTorch!
    [32mINFO    [0m [34mStarting Fine Tuning for 2 epochs...[0m
    I0318 15:34:09.568219 140155325249344 fine_tune.py:101] Starting Fine Tuning for 2 epochs...
    I0318 15:34:09.647709 140155325249344 rank_zero.py:64] GPU available: True (cuda), used: True
    I0318 15:34:09.662323 140155325249344 rank_zero.py:64] TPU available: False, using: 0 TPU cores
    I0318 15:34:09.662955 140155325249344 rank_zero.py:64] IPU available: False, using: 0 IPUs
    I0318 15:34:09.663497 140155325249344 rank_zero.py:64] HPU available: False, using: 0 HPUs
    I0318 15:34:09.669328 140155325249344 rank_zero.py:64] You are using a CUDA device ('NVIDIA GeForce RTX 3070') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision


    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified


    I0318 15:34:13.796578 140155325249344 cuda.py:61] LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    I0318 15:34:13.808069 140155325249344 model_summary.py:94] 
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


    Sanity Checking DataLoader 0:  50%|█████     | 1/2 [00:00<00:00, 13.05it/s]

    /opt/conda/envs/mase/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:441: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=31` in the `DataLoader` to improve performance.


                                                                               

    /opt/conda/envs/mase/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=31` in the `DataLoader` to improve performance.


    Epoch 1: 100%|██████████| 196/196 [00:43<00:00,  4.55it/s, v_num=8, train_acc_step=0.863, val_acc_epoch=0.893, val_loss_epoch=0.223]

    I0318 15:35:42.103955 140155325249344 rank_zero.py:64] `Trainer.fit` stopped: `max_epochs=2` reached.


    Epoch 1: 100%|██████████| 196/196 [00:43<00:00,  4.52it/s, v_num=8, train_acc_step=0.863, val_acc_epoch=0.893, val_loss_epoch=0.223]

    [32mINFO    [0m [34mFine Tuning Complete[0m
    I0318 15:35:42.213792 140155325249344 fine_tune.py:120] Fine Tuning Complete
    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0318 15:35:42.219084 140155325249344 quantize.py:129] Converting PyTorch model to ONNX...


    


    /opt/conda/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:363: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax < 0:
    /opt/conda/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:366: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      max_bound = torch.tensor((2.0**(num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    /opt/conda/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:376: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
    /opt/conda/envs/mase/lib/python3.11/site-packages/pytorch_quantization/tensor_quant.py:382: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if min_amax <= epsilon:
    [32mINFO    [0m [34mONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/onnx/2024_03_18/version_15/model.onnx[0m
    I0318 15:35:43.000104 140155325249344 quantize.py:152] ONNX Conversion Complete. Stored ONNX model to /root/mase/mase_output/tensorrt/quantization/onnx/2024_03_18/version_15/model.onnx
    [32mINFO    [0m [34mConverting PyTorch model to TensorRT...[0m
    I0318 15:35:43.003496 140155325249344 quantize.py:55] Converting PyTorch model to TensorRT...


    [03/18/2024-15:35:48] [TRT] [W] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.


    [32mINFO    [0m [34mTensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/trt/2024_03_18/version_10/model.trt[0m
    I0318 15:38:52.261729 140155325249344 quantize.py:124] TensorRT Conversion Complete. Stored trt model to /root/mase/mase_output/tensorrt/quantization/trt/2024_03_18/version_10/model.trt
    [32mINFO    [0m [34mTensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/json/2024_03_18/version_10/model.json[0m
    I0318 15:38:52.553308 140155325249344 quantize.py:168] TensorRT Model Summary Exported to /root/mase/mase_output/tensorrt/quantization/json/2024_03_18/version_10/model.json



```python
_, _ = runtime_analysis_pass(meta['trt_engine_path'], pass_args=runtime_analysis_config)
```

    [32mINFO    [0m [34m
    TensorRT Engine Input/Output Information:
    Index | Type    | DataType | Static Shape         | Dynamic Shape        | Name
    ------|---------|----------|----------------------|----------------------|-----------------------
    0     | Input   | FLOAT    | (256, 3, 32, 32)       | (256, 3, 32, 32)       | input
    1     | Output  | FLOAT    | (256, 10)              | (256, 10)              | 220[0m
    I0318 15:38:52.609910 140155325249344 analysis.py:117] 
    TensorRT Engine Input/Output Information:
    Index | Type    | DataType | Static Shape         | Dynamic Shape        | Name
    ------|---------|----------|----------------------|----------------------|-----------------------
    0     | Input   | FLOAT    | (256, 3, 32, 32)       | (256, 3, 32, 32)       | input
    1     | Output  | FLOAT    | (256, 10)              | (256, 10)              | 220
    [32mINFO    [0m [34mStarting transformation analysis[0m
    I0318 15:38:52.611444 140155325249344 analysis.py:214] Starting transformation analysis


    [32mINFO    [0m [34m
    Results vgg7-quantized:
    +------------------------------+-----------+
    |            Metric            |   Value   |
    +------------------------------+-----------+
    |    Average Test Accuracy     |  0.93222  |
    |      Average Precision       |  0.93206  |
    |        Average Recall        |  0.93222  |
    |       Average F1 Score       |  0.93204  |
    |         Average Loss         |  0.22506  |
    |       Average Latency        | 19.479 ms |
    |   Average GPU Power Usage    | 110.15 W  |
    | Inference Energy Consumption | 0.596 mWh |
    +------------------------------+-----------+[0m
    I0318 15:38:55.434816 140155325249344 analysis.py:323] 
    Results vgg7-quantized:
    +------------------------------+-----------+
    |            Metric            |   Value   |
    +------------------------------+-----------+
    |    Average Test Accuracy     |  0.93222  |
    |      Average Precision       |  0.93206  |
    |        Average Recall        |  0.93222  |
    |       Average F1 Score       |  0.93204  |
    |         Average Loss         |  0.22506  |
    |       Average Latency        | 19.479 ms |
    |   Average GPU Power Usage    | 110.15 W  |
    | Inference Energy Consumption | 0.596 mWh |
    +------------------------------+-----------+


## Section 5. Language Models


```python
# Path to your TOML file
toml_file_path = '../../../machop/configs/tensorrt/opt-125M_layerwise_mixed_precision_by_name.toml'

# Reading TOML file and converting it into a Python dictionary
with open(toml_file_path, 'r') as toml_file:
    pass_args = toml.load(toml_file)

# Extract the 'passes.tensorrt' section and its children
tensorrt_config = pass_args.get('passes', {}).get('tensorrt', {})
# Extract the 'passes.runtime_analysis' section and its children
runtime_analysis_config = pass_args.get('passes', {}).get('runtime_analysis', {})

# Load the basics in
model_name = pass_args['model']
dataset_name = pass_args['dataset']
max_epochs = pass_args['max_epochs']
batch_size = pass_args['batch_size']
learning_rate = pass_args['learning_rate']
accelerator = pass_args['accelerator']

opt_tokenizer = get_tokenizer("facebook/opt-125m")

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0, # os.cpu_count()
    max_token_len=128,
    tokenizer=opt_tokenizer,
    load_from_cache_file=True,
)
data_module.prepare_data()
data_module.setup()

# Add the data_module and other necessary information to the configs
configs = [tensorrt_config, runtime_analysis_config]
for config in configs:
    config['task'] = pass_args['task']
    config['batch_size'] = pass_args['batch_size']
    config['model'] = pass_args['model']
    config['data_module'] = data_module
    config['accelerator'] = 'cuda' if pass_args['accelerator'] == 'gpu' else pass_args['accelerator']
    if config['accelerator'] == 'gpu':
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
```


```python
model_info = get_model_info(model_name)
model = get_model(
    "facebook/opt-125m:patched",
    task="lm",
    dataset_info=get_dataset_info("wikitext2"),
    pretrained=True,
)

# Load in the trained checkpoint - change this accordingly
# OPT125M_CHECKPOINT_PATH = "../../../mase_output/jsc-toy_classification_jsc_2024-03-17/software/training_ckpts/best.ckpt"
# model = load_model(load_name=OPT125M_CHECKPOINT_PATH, load_type="pl", model=model)

model_info = get_model_info("facebook/opt-125m:patched")
cf_args = get_cf_args(model_info=model_info, task="lm", model=model)

mg = MaseGraph(model=model, cf_args=cf_args)

# dummy_in = get_dummy_input(model_info, data_module=data_module, task="lm")
# if len(mg.model.additional_inputs) > 0:
#     dummy_in = dummy_in | mg.model.additional_inputs

# Initiate metadata
mg, _ = init_metadata_analysis_pass(mg, pass_args=None)

# # Before we begin, we will copy the original MaseGraph model to use for comparison during quantization analysis
# mg_original = deepcopy_mase_graph(mg)
```


```python
# _, _ = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)
```


```python
# mg, _ = tensorrt_fake_quantize_transform_pass(mg, pass_args=tensorrt_config)
# summarize_quantization_analysis_pass(mg_original, mg)

# mg, _ = tensorrt_calibrate_transform_pass(mg, pass_args=tensorrt_config)

# mg, _ = tensorrt_fine_tune_transform_pass(mg, pass_args=tensorrt_config)

mg, meta = tensorrt_engine_interface_pass(mg, pass_args=tensorrt_config)

_, _ = runtime_analysis_pass(mg_original, pass_args=runtime_analysis_config)
_, _ = runtime_analysis_pass(meta['trt_engine_path'], pass_args=runtime_analysis_config)
```

    [32mINFO    [0m [34mConverting PyTorch model to ONNX...[0m
    I0318 08:54:47.010779 139760749061952 quantize.py:129] Converting PyTorch model to ONNX...
    ERROR:tornado.general:SEND Error: Host unreachable
    Traceback (most recent call last):
      File "_pydevd_bundle/pydevd_cython.pyx", line 577, in _pydevd_bundle.pydevd_cython.PyDBFrame._handle_exception
      File "_pydevd_bundle/pydevd_cython.pyx", line 312, in _pydevd_bundle.pydevd_cython.PyDBFrame.do_wait_suspend
      File "/opt/conda/envs/mase/lib/python3.11/site-packages/debugpy/_vendored/pydevd/pydevd.py", line 2070, in do_wait_suspend
        keep_suspended = self._do_wait_suspend(thread, frame, event, arg, suspend_type, from_this_thread, frames_tracker)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/envs/mase/lib/python3.11/site-packages/debugpy/_vendored/pydevd/pydevd.py", line 2106, in _do_wait_suspend
        time.sleep(0.01)
    KeyboardInterrupt

