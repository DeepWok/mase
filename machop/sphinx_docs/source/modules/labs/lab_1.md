<!-- # Lab 1 for Advanced Deep Learning Systems (ADLS ELEC70109/EE9-AML3-10/EE9-AO25) -->

<br />
<div align="center">
  <a href="https://deepwok.github.io/">
    <img src="../imgs/deepwok.png" alt="Logo" width="160" height="160">
  </a>

  <h1 align="center">Lab 1 for Advanced Deep Learning Systems (ADLS)</h1>

  <p align="center">
    ELEC70109/EE9-AML3-10/EE9-AO25
    <br />
  Written by
    <a href="https://aaron-zhao123.github.io/">Aaron Zhao </a> ,
    <a href="https://chengzhang-98.github.io/blog/">Cheng Zhang </a> ,
    <a href="https://www.pedrogimenes.co.uk/">Pedro Gimenes </a>
  </p>
</div>

# General introduction

In this lab, you will learn how to use the basic functionalities of the software stack of MASE.

There are in total 5 tasks you would need to finish.

# Preparation and installations

Make sure you have read and understood the installation of the framework in [Get-started-on-local-machines-software-only](../Get-started-on-local-machines-software-only.md).

Both streams (software and hardware) would start from software first. We are starting with simple toy networks that can train on most laptops.

The recommendation is to fork this repository to your own Github account. If you are not familiar with `git`, `github` and terminologies like `fork`. You might find this [webpage](https://docs.github.com/en/get-started/quickstart/fork-a-repo) useful. You might also want to work on your own [`branch`](<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches>).

Please be aware that pushing to the `main` branch would be blocked, and your final project would be submitted as a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

# Getting familiar with the Machop Command Line

The software framework of MASE is called Machop, you can find a reason for the naming in the Machop [Readme](../../machop/README.md).

After installation, you should be able to run

```bash
./ch --help
```

You should see a print out of a list of options and a usage guide, this is also a good test for your installation. If your installation is incorrect, this command will throw you an error.

The print out has the following lines:

```bash
main arguments:
  action                action to perform. One of (train|test|transform|search)
  model                 name of a supported model. Required if configuration NOT provided.
  dataset               name of a supported dataset. Required if configuration NOT provided.
```

This means the `ch` command line tool expects three compulsory inputs that are `action`, `model` and `dataset` respectively.
These three components would have to be either defined in the command line interface or be defined in a configuration file.

```bash
# example
# train is an action, mnist is the dataset and toynet is the model name.
# you do not have to run the command for now.
./ch train toy mnist
```

The command line interface also allows you to input additional arguments to control the training and testing flow.

```bash
# example
# setting the maximum training epochs and batch size through the cmd line interface.
# you do not have to run the command for now.
./ch train toy mnist --max-epochs 200 --batch-size 256
```

# Training your first network

In this section, we are interested in training a small network and evaluate the trained network through the command line flow.

The dataset we look at is the Jet Substructure Classification (JSC) dataset.

> [A bit of physics]
Jets are collimated showers of particles that result from the decay and hadronization of quarks q and gluons g.
At the Large Hadron Collider (LHC), due to the high collision energy, a particularly interesting jet signature emerges from overlapping quark-initiated showers produced in decays of heavy standard model particles.
It is the task of jet substructure to distinguish the various radiation profiles of these jets from backgrounds consisting mainly of quark (u, d, c, s, b) and gluon-initiated jets. The tools of jet substructure have been used to distinguish interesting jet signatures from backgrounds that have production rates hundreds of times larger than the signal.

In short, the dataset contains inputs with a feature size of 16 and 5 output classes.

## The train command

To train a network for the JSC dataset, you would need to run:

```bash
# You will need to run this command
./ch train jsc-tiny jsc --max-epochs 10 --batch-size 256
```

`--max-epochs` states the maximum epochs allowed to train, and `--batch-size` defines the batch size for training.

You should see a print out of the training configuration in a table

```bash
+-------------------------+--------------------------+-----------------+--------------------------+
| Name                    |         Default          | Manual Override |        Effective         |
+-------------------------+--------------------------+-----------------+--------------------------+
| task                    |      classification      |                 |      classification      |
| load_name               |           None           |                 |           None           |
| load_type               |            mz            |                 |            mz            |
| batch_size              |           128            |       256       |           256            |
| to_debug                |          False           |                 |          False           |
| log_level               |           info           |                 |           info           |
| seed                    |            0             |                 |            0             |
| training_optimizer      |           adam           |                 |           adam           |
| trainer_precision       |            32            |                 |            32            |
| learning_rate           |          1e-05           |                 |          1e-05           |
| weight_decay            |            0             |                 |            0             |
| max_epochs              |            20            |       10        |            10            |
| max_steps               |            -1            |                 |            -1            |
| accumulate_grad_batches |            1             |                 |            1             |
| log_every_n_steps       |            50            |                 |            50            |
| num_workers             |            16            |        0        |            0             |
| num_devices             |            1             |                 |            1             |
| num_nodes               |            1             |                 |            1             |
| accelerator             |           auto           |                 |           auto           |
| strategy                |           ddp            |                 |           ddp            |
| is_to_auto_requeue      |          False           |                 |          False           |
| github_ci               |          False           |                 |          False           |
| disable_dataset_cache   |          False           |                 |          False           |
| target                  |   xcu250-figd2104-2L-e   |                 |   xcu250-figd2104-2L-e   |
| num_targets             |           100            |                 |           100            |
| is_pretrained           |          False           |                 |          False           |
| max_token_len           |           512            |                 |           512            |
| project_dir             | /Users/aaron/Projects/ma |                 | /Users/aaron/Projects/ma |
|                         |   se-tools/mase_output   |                 |   se-tools/mase_output   |
| project                 |           None           |                 |           None           |
| model                   |           None           |    jsc-tiny     |         jsc-tiny         |
| dataset                 |           None           |       jsc       |           jsc            |
+-------------------------+--------------------------+-----------------+--------------------------+
```

There is also a summary on the model

```bash
  | Name      | Type               | Params
-------------------------------------------------
0 | model     | JSC_Tiny           | 127
1 | loss_fn   | CrossEntropyLoss   | 0
2 | acc_train | MulticlassAccuracy | 0
3 | acc_val   | MulticlassAccuracy | 0
4 | acc_test  | MulticlassAccuracy | 0
5 | loss_val  | MeanMetric         | 0
6 | loss_test | MeanMetric         | 0
-------------------------------------------------
127       Trainable params
0         Non-trainable params
127       Total params
0.001     Total estimated model params size (MB)
```

## Logging on tensorboard

As you can see, this is a toy model and it is very small. There is another print out line that is also very useful:

```bash
Project will be created at /home/cheng/GTA/adls/mase-tools/mase_output/jsc-tiny_classification_jsc_2023-10-30
Missing logger folder: /Users/aaron/Projects/mase-tools/mase_output/jsc-tiny_classification_jsc_2023-10-19/software/training_ckpts/logs
```

For any training commands executed, a logging directory would be created and one can use [tensorboard](https://www.tensorflow.org/tensorboard) to check the training trajectory.

```bash
# You need to run the following command with your own edits

# the actual name of the log file created for you would be different from the example showing here, because the name contains a time-stamp.

# --port 16006 is declaring the port on localhost
tensorboard --logdir ../mase_output/jsc-tiny_classification_jsc_2023-10-19/software --port 16006
```

Open `http://localhost:16006/` in your preferred browser, explore on the entries that have been logged.

## The test command

Under the same folder
`../mase_output/jsc-tiny_classification_jsc_2023-10-19/software`, there are also saved checkpoint files for the trained models. These are basically the trained parameters of the model, one can find more detail on Pytorch model checkpointing [here](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html) and Lightning checkpointing [here](https://lightning.ai/docs/pytorch/stable/common/checkpointing.html).

```bash
./ch test jsc-tiny jsc --load ../mase_output/jsc-tiny_classification_jsc_2023-10-19/software/training_ckpts/best.ckpt --load-type pl
```

The above command would return you the performance of the trained model on the test set. `--load-type pl` tells Machop that the checkpoint is saved by PyTorch Lightning. For PyTorch Lightning, see [this section](#the-entry-point-for-the-train-action)

The saved checkpoint can also be used to resume training.

## The definition of the JSC dataset

Datasets are defined in under the [dataset](../../machop/chop/dataset) folder in `chop`, one should take a look at the [\_\_init__.py](../../machop/chop/dataset/__init__.py) to understand how different datasets are declared. The JSC dataset is defined and detailed in [this file](../../machop/chop/dataset/physical/jsc.py#L142):

```python
@add_dataset_info(
    name="jsc",
    dataset_source="manual",
    available_splits=("train", "validation", "test"),
    physical_data_point_classification=True,
    num_classes=5,
    num_features=16,
)
class JetSubstructureDataset(Dataset):
    def __init__(self, input_file, config_file, split="train"):
        super().__init__()
  ...
```

The [decorator](https://book.pythontips.com/en/latest/decorators.html) (if you do not know what is a python decorator, click the link and learn) defines the dataset information required. The class object `JetSubstructureDataset` has `Dataset` being its parent class. If you are still concerned with your proficiency in OOP (object orientated programming), you should check this [link](https://book.pythontips.com/en/latest/classes.html).

## The definition of the JSC Tiny network

The network definition can also be found in the [\_\_init__.py](../../machop/chop/models/physical/jet_substructure/__init__.py#32)

```python
class JSC_Tiny(nn.Module):
    def __init__(self, info):
        super(JSC_Tiny, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  #  batch norm layer
            nn.Linear(16, 5),  # linear layer
        )

    def forward(self, x):
        return self.seq_blocks(x)
```

Network definitions in Pytorch normally contains two components: an `__init__` method and a `forward` method. Also all networks and custom layers in Pytorch has to be a subclass of  `nn.Module`.
The neural network layers are initialised in `__init__`. Every `nn.Module` subclass implements the operations on input data in the `forward` method.

`nn.Sequential` is a container used for wrapping a number of layers together, more information on this container can be found in this [link](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html).

# Varying the parameters

We have executed the following training command:

````bash
./ch train jsc-tiny jsc --max-epochs 10 --batch-size 256
````

We can, apparently, tune a bunch of parameters, and the obvious ones to tune are

* `batch-size`
* `max-epochs`
* `learning-rate`

Tune these parameters by hand and answer the following questions:

1. What is the impact of varying batch sizes and why?
2. What is the impact of varying maximum epoch number?
3. What is happening with a large learning and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?

# A deeper dive into the framework

When you execute `./ch`, what really happens is the [ch](../../machop/ch) file got executed and from the `import` you can tell it is calling into [cli.py](../../machop/chop/cli.py).

## The entry point for the train/teset action

When you choose to execute `./ch train`, we are executing the train action, and invoking [train.py](../../machop/chop/actions/train.py). The entire training flow is orchestrated using [PyTorch Lightning](https://lightning.ai/), so that the detailed lightning related wrapping occurs in [jet_substructure.py](../../machop/chop/plt_wrapper/physical/jet_substructure.py). PyTorch Lightning's checkpointing callbacks saves the model parameters (`torch.nn.Module.state_dict()`), the optimizer states, and other hyper-parameters specified in `lightning.pl.LightningModule`, so that the training can be resumed from the last checkpoint. The saved checkpoint has extension `.ckpt`, this is why we have `--load-type pl` in the `./ch test` command.

Test action has similar implementation based on PyTorch Lightning ([test.py](../../machop/chop/actions/test.py))

## The entry point for the model

All models are defined in the [\_\_init__.py](../../machop/chop/models/__init__.py) under the model folder. The `get_model` function is called inside `actions` (such as `train`) to ping down different models.

## The entry point for the dataset

Similar to the model definitions, all datasets are defined in the [\_\_init__.py](../../machop/chop/dataset/__init__.py) under the dataset folder.

# Train your own network

Now you are familiar with different components in the tool.

4. Implement a network that has in total around 10x more parameters than the toy network.
5. Test your implementation and evaluate its performance.


# Google Colab Adaption

[lab1.ipynb](./lab1.ipynb) contains an adaption of setting up the same thing on Google Colab. You would need to repeat the exercise on that because you would definitely need a powerful GPU for later labs and your Team Projects.