# Advanced: Using Mase CLI

The Mase CLI has been deprecated, but the following tutorials are kept here for legacy reasons.

## Run the train action with the CLI

MASE has several functionalities, and this document aims to introduce the simplest `train` and `eval` pipelines.

### Command line interface

MASE actually supports usage in two modes:
* A direct `import` as a module (eg. `machop/examples/toy/main.py`).
* Through the command line interface (the focus of this document).

In this case, we can try a toymodel, the command looks like the following

```bash
# assuming you you are at the our-stuff/mase directory
cd src 
./ch train toy toy_tiny --config ../configs/archive/test/train.toml --max-epochs 3
```

> **Note:** This is training is for demonstration purposes, we picked a very small model/dataset to make it runnable even on CPU-based devices. It does not mean to achieve a useful accuracy!

You can fetch all command-line arguments:

```bash
[nix-shell:~/Projects/mase/src]$ ./ch -help
INFO     Set logging level to debug
WARNING  TensorRT pass is unavailable because the following dependencies are not installed: pytorch_quantization, tensorrt, pycuda, cuda.
usage: ch [--config PATH] [--task TASK] [--load PATH] [--load-type] [--batch-size NUM] [--debug] [--log-level] [--report-to {wandb,tensorboard}] [--seed NUM] [--quant-config TOML]
          [--training-optimizer TYPE] [--trainer-precision TYPE] [--learning-rate NUM] [--weight-decay NUM] [--max-epochs NUM] [--max-steps NUM] [--accumulate-grad-batches NUM]
          [--log-every-n-steps NUM] [--cpu NUM] [--gpu NUM] [--nodes NUM] [--accelerator TYPE] [--strategy TYPE] [--auto-requeue] [--github-ci] [--disable-dataset-cache]
          [--target STR] [--num-targets NUM] [--run-emit] [--skip-build] [--skip-test] [--pretrained] [--max-token-len NUM] [--project-dir DIR] [--project NAME] [--profile]
          [--no-warnings] [-h] [-V] [--info [TYPE]]
          action [model] [dataset]

Chop is a simple utility, part of the MASE tookit, to train, test and transform (i.e. prune or quantise) a supported model.

main arguments:
  action                action to perform. One of (train|test|transform|search|emit|simulate)
  model                 name of a supported model. Required if configuration NOT provided.
  dataset               name of a supported dataset. Required if configuration NOT provided.

general options:
  --config PATH         path to a configuration file in the TOML format. Manual CLI overrides for arguments have a higher precedence. Required if the action is transform. (default:
                        None)
  --task TASK           task to perform. One of (classification|cls|translation|tran|language_modeling|lm) (default: classification)
  --load PATH           path to load the model from. (default: None)
  --load-type           the type of checkpoint to be loaded; it's disregarded if --load is NOT specified. It is designed to and must be used in tandem with --load. One of
                        (pt|pl|mz|hf) (default: mz)
  --batch-size NUM      batch size for training and evaluation. (default: 128)
  --debug               run the action in debug mode, which enables verbose logging, custom exception hook that uses ipdb, and sets the PL trainer to run in "fast_dev_run" mode.
                        (default: False)
  --log-level           verbosity level of the logger; it's only effective when --debug flag is NOT passed in. One of (debug|info|warning|error|critical) (default: info)
  --report-to {wandb,tensorboard}
                        reporting tool for logging metrics. One of (wandb|tensorboard) (default: tensorboard)
  --seed NUM            seed for random number generators set via Pytorch Lightning's seed_everything function. (default: 0)
  --quant-config TOML   path to a configuration file in the TOML format. Manual CLI overrides for arguments have a higher precedence. (default: None)

trainer options:
  --training-optimizer TYPE
                        name of supported optimiser for training. One of (adam|sgd|adamw) (default: adam)
  --trainer-precision TYPE
                        numeric precision for training. One of (16-mixed|32|64|bf16) (default: 16-mixed)
  --learning-rate NUM   initial learning rate for training. (default: 1e-05)
  --weight-decay NUM    weight decay for training. (default: 0)
  --max-epochs NUM      maximum number of epochs for training. (default: 20)
  --max-steps NUM       maximum number of steps for training. A negative value disables this option. (default: -1)
  --accumulate-grad-batches NUM
                        number of batches to accumulate gradients. (default: 1)
  --log-every-n-steps NUM
                        log every n steps. No logs if num_batches < log_every_n_steps. (default: 50))

runtime environment options:
  --cpu NUM, --num-workers NUM
                        number of CPU workers; the default varies across systems and is set to os.cpu_count(). (default: 12)
  --gpu NUM, --num-devices NUM
                        number of GPU devices. (default: 1)
  --nodes NUM           number of nodes. (default: 1)
  --accelerator TYPE    type of accelerator for training. One of (auto|cpu|gpu|mps) (default: auto)
  --strategy TYPE       type of strategy for training. One of (auto|ddp|ddp_find_unused_parameters_true) (default: auto)
  --auto-requeue        enable automatic job resubmission on SLURM managed cluster. (default: False)
  --github-ci           set the execution environment to GitHub's CI pipeline; it's used in the MASE verilog emitter transform pass to skip simulations. (default: False)
  --disable-dataset-cache
                        disable caching of datasets. (default: False)

hardware generation options:
  --target STR          target FPGA for hardware synthesis. (default: xcu250-figd2104-2L-e)
  --num-targets NUM     number of FPGA devices. (default: 100)
  --run-emit
  --skip-build
  --skip-test

language model options:
  --pretrained          load pretrained checkpoint from HuggingFace/Torchvision when initialising models. (default: False)
  --max-token-len NUM   maximum number of tokens. A negative value will use tokenizer.model_max_length. (default: 512)

project options:
  --project-dir DIR     directory to save the project to. (default: /Users/yz10513/Projects/mase/mase_output)
  --project NAME        name of the project. (default: {MODEL-NAME}_{TASK-TYPE}_{DATASET-NAME}_{TIMESTAMP})
  --profile
  --no-warnings

information:
  -h, --help            show this help message and exit
  -V, --version         show version and exit
  --info [TYPE]         list information about supported models or/and datasets and exit. One of (all|model|dataset) (default: all)

Maintained by the DeepWok Lab. Raise issues at https://github.com/JianyiCheng/mase-tools/issues
(.venv)
[nix-shell:~/Projects/mase/src]$
```

### Training implementation

The core train file is `mase/src/chop/actions/train.py`, and the model is wrapped using `torch_lightning`(<https://www.pytorchlightning.ai/index.html>). These wrappers are different for various tasks, as you can see in `src/chop/tools/plt_wrapper/__init__.py`:

```python
def get_model_wrapper(model_info, task: str):
    if model_info.is_physical_model:
        return JetSubstructureModelWrapper
    elif model_info.is_nerf_model:
        return NeRFModelWrapper
    elif model_info.is_vision_model:
        return VisionModelWrapper
    elif model_info.is_nlp_model:
        match task:
            case "classification" | "cls":
                return NLPClassificationModelWrapper
            case "language_modeling" | "lm":
                return NLPLanguageModelingModelWrapper
            case "translation" | "tran":
                return NLPTranslationModelWrapper
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
```

This is because different tasks may have a different `forward` pass and also might use different manipulations on data and also its evaluation metrics.
These `Wrapper` also defines the detailed `training_flow`, if we use `src/chop/tools/splt_wrapper/base.py` as an example, `training_step` and `validation_step` are defined in this class.

```bash

class WrapperBase(pl.LightningModule):
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.acc_train(y_hat, y)
        self.log("train_acc_step", self.acc_train, prog_bar=True)
        self.log("train_loss_step", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
				...
```

### Models

Models are mainly implemented in `src/chop/models/__init__.py
`, and the `get_model_info` contains all currently supported models.
One can also instantiate a custom model and add it to our flow like `src/chop/models/toys`.

### Output

MASE produces an output directory after running the training flow. The output directory is found at `../mase_output/<model>_<task>_<dataset>_<current_date>`.
This directory includes

* `hardware` - a directory for Verilog hardware generated for the trained model

* `software` - a directory for any software generated for the trained model (PyTorch checkpoints or MASE models) as well as any generated logs

### Training Logs

MASE creates [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) or [wandb](https://wandb.ai/site) logs for the training flow - allowing tracking and visualizing metrics such as loss and accuracy. The log files are in `<output_dir>/software/tensorboard/lightning_logs/version_<n>`. 

Run Tensorboard to visualise the logs using:

```bash
tensorboard --logdir <path_to_log_files>
```

If you are using VSCode, this will show up popup asking if you want to open Tensorboard in your browser. Select yes.

If you look at the training printouts, you should have seen something like the following

```bash
INFO     Initialising model 'toy'...
INFO     Initialising dataset 'toy_tiny'...
INFO     Project will be created at ../mase_output/toy_classification_toy_tiny_2024-06-13
INFO     Training model 'toy'...
```

This means the `path_to_log_files` is `../mase_output/toy_classification_toy_tiny_2024-06-13`.

Full command should be something like

```bash
tensorboard --logdir ../mase_output/toy_classification_toy_tiny_2024-06-13/software/tensorboard/lightning_logs/version_2
```

### Test command

To test the model trained above you can use:

```bash
# After training, you will have your checkpoint under mase-tools/mase_output
# For example, the checkpoint is under ../mase_output/toy_classification_toy-tiny_2023-07-03/software/training_ckpts/best.ckpt 
./ch test toy toy_tiny --config ../configs/archive/test/train.toml --load ../mase_output/toy_classification_toy_tiny_2024-06-13/software/training_ckpts/best.ckpt```


## Run the transform action with the CLI

### Train a model

```bash
./ch train toy toy_tiny --config ../configs/archive/test/train.toml --max-epochs 3
```

### Quantise transform

- The above command should have automatically saved a checkpoint file (`your_post_training_checkpoint`) for you, which was `../mase_output/toy_classification_toy_tiny_2024-06-20/software/training_ckpts/best.ckpt`, you should use these generated checkpoint files for later command line instructions.
- Quantise it with fixed-point quantisers
- The load path `--load` changes with your generation time of course

The config takes the following format:

```toml
# basics
model = "toy"
dataset = "toy-tiny"

[passes.quantize]
by = "type"
report = true

[passes.quantize.default.config]
name = "integer"
data_in_width = 8
data_in_frac_width = 4
weight_width = 8
weight_frac_width = 9
bias_width = 8
bias_frac_width = 9
```

With the config, we can quantise the whole network to integer arithmetic by doing:

```bash
./ch transform toy toy_tiny --config ../configs/examples/toy_uniform.toml --load your_post_training_checkpoint --load-type pl
```

> **_NOTE:_** the file name `your_post_training_checkpoint` is subject to change, this should be the checkpoint file generated from the training action.

### Quantization-aware Training (QAT)

- Load and do Quantisation-aware-training (QAT) with a transformed model

```bash
./ch train --config ../configs/examples/toy_uniform.toml --model toy --load your_post_transform_mz --load-type mz  --max-epochs 3
```

- Quantise it with fixed-point quantisers
- The load path `--load` changes with your generation time of course

> **_NOTE:_** `transform` by default saves transformed models in a mase format called `xxx.mz`, fine-tuning the transformed model only need to load this `.mz` format.

### Quantise transform by Type

```toml
# basics
model = "toy"
dataset = "toy-tiny"

[passes.quantize]
by = "type"
report = true

[passes.quantize.default.config]
name = "NA"

[passes.quantize.linear.config]
name = "integer"
data_in_width = 8
data_in_frac_width = 4
weight_width = 8
weight_frac_width = 9
bias_width = 8
bias_frac_width = 9
```

We recommend you to take a look at the configuration file at `mase/configs/examples/toy_uniform.toml`. In this case, only linear layers are quantised!

## Mixed-precision search on manual model

This tutorial shows how to search for mixed-precision quantization strategy for OPT model on Wikitext2 dataset.

> **Note**: Manual model refers to the model named as `<model_arch>_quantized` at `mase-tools/machop/chop/models/manual`. Usually these are models that cannot be directly converted to MASE Graph.

### Search for Mixed-Precision Quantization Scheme

What is included in this search:
- The checkpoint "facebook/opt-125m" is loaded from HuggingFace.
- A search space is built for OPT-125M, where each matmul/linear layer operand may have a distinct precision.
- The search is launched. In each trial:
    - A quantization config (`q_config`) is sampled from the search space.
    - The pretrained OPT-125M is quantized with `q_config`
    - Software runner evaluates the quantized OPT and return some metrics. In this example, the perplexity on WikiText2 is returned.
    - Hardware runner evaluates the quantized OPT and return some metrics. In this example, the average bitwidth is returned.
    - The trial objective is calculated.

 and search for fixed-point precision on Wikitext2 dataset.

#### Search config

Here is the search part in `configs/examples/search_opt_quantized_tpe_search.toml` looks like the following.

```toml
[search.search_space]
# the search space name defined in mase
name = "module/manual_hf/quantize/llm_mixed_precision_ptq"

[search.search_space.setup]
model_parallel = false

[search.search_space.seed.default]
# Since we are doing mixed-precision search.
# Only one "name" is allowed (len(name) == 1)
name = ["integer"]
# precision search space is specified using the following lists
data_in_width = [2, 4, 8, 10]
data_in_frac_width = [2, 4, 6]
weight_width = [2, 4, 8, 10]
weight_frac_width = [2, 4, 6]
bias_width = [2, 4, 8, 10]
bias_frac_width = [2, 4, 6]

[search.strategy]
name = "optuna"
eval_mode = true

# software (sw) runner and hardware (hw) runner evaluates the quantized model to guide the search
# here we evaluate the perplexity and average bitwidth of the quantized model
[search.strategy.sw_runner.basic_evaluation]
data_loader = "val_dataloader"
num_samples = 512

[search.strategy.hw_runner.average_bitwidth]
compare_to = 32 # compare to FP32

[search.strategy.setup]
# evaluating perplexity requires GPUs so we only launch 1 job.
n_jobs = 1
# we run 10 trials in total for demostration.
n_trials = 10
timeout = 20000
# Optuna supports a range of search algorithms, including Random, TPE, Genetic, etc.
sampler = "TPE"
model_parallel = false
sum_scaled_metrics = false # false for multi-objective, true for single objecive

[search.strategy.metrics]
perplexity.scale = 1.0
perplexity.direction = "minimize"
average_bitwidth.scale = 1.0
average_bitwidth.direction = "minimize"
```

#### Launch the Precision Search

Run the search:
```bash
cd machop
./ch search --config ../configs/examples/search_opt_quantized_tpe_search.toml
```

When the search is done, the best quantization config will be printed out. Since we run multi-objective search. There may be multiple best trials found by Optuna.
```txt
Best trial(s):
|    |   number | software_metrics                     | hardware_metrics                                     | scaled_metrics                                  |
|----+----------+--------------------------------------+------------------------------------------------------+-------------------------------------------------|
|  0 |        0 | {'loss': 12.43, 'perplexity': 6.13}  | {'average_bitwidth': 7.194, 'memory_density': 4.448} | {'average_bitwidth': 7.194, 'perplexity': 6.13} |
|  1 |        2 | {'loss': 11.0, 'perplexity': 21.102} | {'average_bitwidth': 6.0, 'memory_density': 5.333}   | {'average_bitwidth': 6.0, 'perplexity': 21.102} |
```

Usually the TPE can optimize the average bitwidth and perplexity trade-off.

#### Search Logs

The complete search results will be saved in `mase/mase_output/opt_quantized_wikitext2/software/search_ckpts/log.json`.

Here is part of the `log.json` recording all search details.

For example, `log["0"]["user_attrs_sampled_config"]` is the sampled quantization config of trial 0. Expand it and you will set the precision of each matmul/linear layer's operands.

```json
{
    "0":{
        "number":0,
        "values_0":0.5,
        "values_1":0.7849056604,
        "user_attrs_hardware_metrics":{
            "average_bitwidth":3.9245283019
        },
        "user_attrs_sampled_config":{
            "seq_blocks_0":{
                "config":{
                    "name":"integer",
                    "data_in_width":4,
                    "data_in_frac_width":4,
                    "weight_width":2,
                    "weight_frac_width":8,
                    "bias_width":2,
                    "bias_frac_width":8
                }
            },
            ...
        },
        "user_attrs_scaled_metrics":{
            "accuracy":0.5,
            "average_bitwidth":0.7849056604
        },
        "user_attrs_software_metrics":{
            "loss":0.6922941208,
            "accuracy":0.5
        },
        "state":"COMPLETE",
        "datetime_start":1694095030315,
        "datetime_complete":1694095031289,
        "duration":974
    },
    "1":{
        "number":1,
        "values_0":0.5747232437,
        "values_1":0.8150943396,
        "user_attrs_hardware_metrics":{
            "average_bitwidth":4.0754716981
        },
        "user_attrs_sampled_config":{
            "seq_blocks_0":{
                "config":{
                    "name":"integer",
                    "data_in_width":8,
                    "data_in_frac_width":3,
                    "weight_width":4,
                    "weight_frac_width":7,
                    "bias_width":4,
                    "bias_frac_width":3
                }
            },
            ...
        },
        "user_attrs_scaled_metrics":{
            "accuracy":0.5747232437,
            "average_bitwidth":0.8150943396
        },
        "user_attrs_software_metrics":{
            "loss":0.6845972538,
            "accuracy":0.5747232437
        },
        "state":"COMPLETE",
        "datetime_start":1694095031290,
        "datetime_complete":1694095032462,
        "duration":1172
    },
    ...
}
```

## Mixed-precision search on MASE Graph

This tutorial shows how to search for mixed-precision quantization strategy for JSC model (a small toy model).

### Commands

First we train a model on the dataset. After training for some epochs, we get a model with some validation accuracy. The checkpoint is saved at an auto-created location. You can refer to [Run the train action with the CLI](../train/simple_train_flow.md) for more detailed explanation.

The reason why we need a pre-trained model is because we would like to do a post-training-quantization (PTQ) search. This means the quantization happens on a pre-trained model. We then use the PTQ accuracy as a proxy signal for our search.


```bash
cd src 
./ch train jsc-tiny jsc --max-epochs 3 --batch-size 256 --accelerator cpu --project tmp --debug --cpu 0
```

- For the interest of time, we do not train this to convergence, apparently one can adjust `--max-epochs` for longer training epochs.
- We choose to train on `cpu` and `--cpu 0` avoids multiprocessing dataloader issues.

```bash
# search command
./ch search --config ../configs/examples/jsc_toy_by_type.toml --task cls --accelerator=cpu --load ../mase_output/tmp/software/training_ckpts/best.ckpt --load-type pl --cpu 0
```

- The line above issues the search with a configuration file, we discuss the configuration in later sections.

```bash
# train searched network
./ch train jsc-tiny jsc --max-epochs 3 --batch-size 256 --accelerator cpu --project tmp --debug --load ../mase_output/jsc-tiny/software/transform/transformed_ckpt/graph_module.mz --load-type mz

# view searched results
cat ../mase_output/jsc-tiny/software/search_ckpts/best.json
```




### Search Config

Here is the search part in `configs/examples/jsc_toy_by_type.toml` looks like the following.

```toml
# basics
model = "jsc-tiny"
dataset = "jsc"
task = "cls"

max_epochs = 5
batch_size = 512
learning_rate = 1e-2
accelerator = "gpu"
project = "jsc-tiny"
seed = 42
log_every_n_steps = 5

[passes.quantize]
by = "type"
[passes.quantize.default.config]
name = "NA"
[passes.quantize.linear.config]
name = "integer"
"data_in_width" = 8
"data_in_frac_width" = 4
"weight_width" = 8
"weight_frac_width" = 4
"bias_width" = 8
"bias_frac_width" = 4

[transform]
style = "graph"


[search.search_space]
name = "graph/quantize/mixed_precision_ptq"

[search.search_space.setup]
by = "name"

[search.search_space.seed.default.config]
# the only choice "NA" is used to indicate that layers are not quantized by default
name = ["NA"]

[search.search_space.seed.linear.config]
# if search.search_space.setup.by = "type", this seed will be used to quantize all torch.nn.Linear/ F.linear
name = ["integer"]
data_in_width = [4, 8]
data_in_frac_width = ["NA"] # "NA" means data_in_frac_width = data_in_width // 2
weight_width = [2, 4, 8]
weight_frac_width = ["NA"]
bias_width = [2, 4, 8]
bias_frac_width = ["NA"]

[search.search_space.seed.seq_blocks_2.config]
# if search.search_space.setup.by = "name", this seed will be used to quantize the mase graph node with name "seq_blocks_2"
name = ["integer"]
data_in_width = [4, 8]
data_in_frac_width = ["NA"]
weight_width = [2, 4, 8]
weight_frac_width = ["NA"]
bias_width = [2, 4, 8]
bias_frac_width = ["NA"]

[search.strategy]
name = "optuna"
eval_mode = true

[search.strategy.sw_runner.basic_evaluation]
data_loader = "val_dataloader"
num_samples = 512

[search.strategy.hw_runner.average_bitwidth]
compare_to = 32 # compare to FP32

[search.strategy.setup]
n_jobs = 1
n_trials = 5
timeout = 20000
sampler = "tpe"
# sum_scaled_metrics = true # single objective
# direction = "maximize"
sum_scaled_metrics = false # multi objective

[search.strategy.metrics]
# loss.scale = 1.0
# loss.direction = "minimize"
accuracy.scale = 1.0
accuracy.direction = "maximize"
average_bitwidth.scale = 0.2
average_bitwidth.direction = "minimize"
```

### Run the search

When the search is completed, we will see the Pareto frontier trials (`sum_scaled_metrics = false`) or the best trials (`sum_scaled_metrics = true`) printed in the terminal.

```text
[2023-09-06 15:54:24][chop.actions.search.strategies.optuna][INFO] Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics            | scaled_metrics                                 |
|----+----------+------------------------------------+-----------------------------+------------------------------------------------|
|  0 |        0 | {'loss': 0.668, 'accuracy': 0.668} | {'average_bitwidth': 2.038} | {'accuracy': 0.668, 'average_bitwidth': 0.408} |
|  1 |        4 | {'loss': 0.526, 'accuracy': 0.729} | {'average_bitwidth': 4.151} | {'accuracy': 0.729, 'average_bitwidth': 0.83}  |
|  2 |        5 | {'loss': 0.55, 'accuracy': 0.691}  | {'average_bitwidth': 2.113} | {'accuracy': 0.691, 'average_bitwidth': 0.423} |
|  3 |       10 | {'loss': 0.542, 'accuracy': 0.691} | {'average_bitwidth': 2.189} | {'accuracy': 0.691, 'average_bitwidth': 0.438} |
|  4 |       13 | {'loss': 0.556, 'accuracy': 0.681} | {'average_bitwidth': 2.075} | {'accuracy': 0.681, 'average_bitwidth': 0.415} |
|  5 |       19 | {'loss': 0.563, 'accuracy': 0.663} | {'average_bitwidth': 2.0}   | {'accuracy': 0.663, 'average_bitwidth': 0.4}   |

```

The entire searching log is saved in `../mase_output/jsc-tiny/software/search_ckpts/log.json`.

Here is part of the `log.json`

```json
{
    "0":{
        "number":0,
        "values_0":0.5,
        "values_1":0.7849056604,
        "user_attrs_hardware_metrics":{
            "average_bitwidth":3.9245283019
        },
        "user_attrs_sampled_config":{
            "seq_blocks_0":{
                "config":{
                    "name":"integer",
                    "data_in_width":4,
                    "data_in_frac_width":4,
                    "weight_width":2,
                    "weight_frac_width":8,
                    "bias_width":2,
                    "bias_frac_width":8
                }
            },
            ...
        },
        "user_attrs_scaled_metrics":{
            "accuracy":0.5,
            "average_bitwidth":0.7849056604
        },
        "user_attrs_software_metrics":{
            "loss":0.6922941208,
            "accuracy":0.5
        },
        "state":"COMPLETE",
        "datetime_start":1694095030315,
        "datetime_complete":1694095031289,
        "duration":974
    },
    "1":{
        "number":1,
        "values_0":0.5747232437,
        "values_1":0.8150943396,
        "user_attrs_hardware_metrics":{
            "average_bitwidth":4.0754716981
        },
        "user_attrs_sampled_config":{
            "seq_blocks_0":{
                "config":{
                    "name":"integer",
                    "data_in_width":8,
                    "data_in_frac_width":3,
                    "weight_width":4,
                    "weight_frac_width":7,
                    "bias_width":4,
                    "bias_frac_width":3
                }
            },
            ...
        },
        "user_attrs_scaled_metrics":{
            "accuracy":0.5747232437,
            "average_bitwidth":0.8150943396
        },
        "user_attrs_software_metrics":{
            "loss":0.6845972538,
            "accuracy":0.5747232437
        },
        "state":"COMPLETE",
        "datetime_start":1694095031290,
        "datetime_complete":1694095032462,
        "duration":1172
    },
    "2":{
        "number":2,
        "values_0":0.5498154759,
        "values_1":0.8,
        "user_attrs_hardware_metrics":{
            "average_bitwidth":4.0
        },
        "user_attrs_sampled_config":{
            "seq_blocks_0":{
                "config":{
                    "name":"integer",
                    "data_in_width":4,
                    "data_in_frac_width":3,
                    "weight_width":2,
                    "weight_frac_width":4,
                    "bias_width":8,
                    "bias_frac_width":4
                }
            },
            ...
        },
        "user_attrs_scaled_metrics":{
            "accuracy":0.5498154759,
            "average_bitwidth":0.8
        },
        "user_attrs_software_metrics":{
            "loss":0.6868978143,
            "accuracy":0.5498154759
        },
        "state":"COMPLETE",
        "datetime_start":1694095032463,
        "datetime_complete":1694095033622,
        "duration":1159
    },
    ...
}
```
