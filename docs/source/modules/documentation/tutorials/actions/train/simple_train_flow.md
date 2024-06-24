# Run the train action with the CLI

MASE has several functionalities, and this document aims to introduce the simplest `train` and `eval` pipelines.

## Command line interface

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

## Training implementation

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

## Models

Models are mainly implemented in `src/chop/models/__init__.py
`, and the `get_model_info` contains all currently supported models.
One can also instantiate a custom model and add it to our flow like `src/chop/models/toys`.

## Output

MASE produces an output directory after running the training flow. The output directory is found at `../mase_output/<model>_<task>_<dataset>_<current_date>`.
This directory includes

* `hardware` - a directory for Verilog hardware generated for the trained model

* `software` - a directory for any software generated for the trained model (PyTorch checkpoints or MASE models) as well as any generated logs

## Training Logs

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

## Test command

To test the model trained above you can use:

```bash
# After training, you will have your checkpoint under mase-tools/mase_output
# For example, the checkpoint is under ../mase_output/toy_classification_toy-tiny_2023-07-03/software/training_ckpts/best.ckpt 
./ch test toy toy_tiny --config ../configs/archive/test/train.toml --load ../mase_output/toy_classification_toy_tiny_2024-06-13/software/training_ckpts/best.ckpt```
