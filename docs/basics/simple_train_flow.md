# Introduction

MASE has several functionalities, and this document aims to introduce the simplest `train` and `eval` pipelines.

# Train command

## Command line interface

MASE actually supports usage in two modes:
* A direct `import` as a module (eg. `machop/examples/toy/main.py`).
* Through the command line interface (the focus of this document).

In this case, we can try a toymodel, the command looks like the following

```bash
cd mase-tools/machop
./ch train toy toy-tiny --config configs/archive/test/train.toml
```

You can fetch all command-line arguments:

```bash
./ch -v
[2023-07-04 13:43:10,177] [INFO] [real_accelerator.py:110:get_accelerator] Setting ds_accelerator to cuda (auto detect)
usage: ch [--config PATH] [--task TASK] [--load PATH] [--load-type]
          [--batch-size NUM] [--debug] [--log-level] [--seed NUM]
          [--training-optimizer TYPE] [--trainer-precision TYPE]
          [--learning-rate NUM] [--max-epochs NUM] [--max-steps NUM]
          [--accumulate-grad-batches NUM] [--cpu NUM] [--gpu NUM]
          [--nodes NUM] [--accelerator TYPE] [--strategy TYPE]
          [--auto-requeue] [--github-ci] [--target STR] [--num-targets NUM]
          [--pretrained] [--max-token-len NUM] [--project-dir DIR]
          [--project NAME] [-h] [-V] [--info [TYPE]]
          action [model] [dataset]

positional arguments:
  action                The action to be performed. Must be one of ['train', 'eval', 'transform', 'search']

options:
  -h, --help            show this help message and exit
  --github-ci           Run in GitHub CI. Default=False
  --debug               Run in debug mode. Default=False
...
```

## Training implementation

The core train file is `machop/chop/actions/train.py`, and the model is wrapped using `torch_lightning`(<https://www.pytorchlightning.ai/index.html>). These wrappers are different for various tasks, as you can see in `machop/chop/plt_wrapper/__init__.py`:

```python
def get_model_wrapper(name, task):
    if name in vision_models:
        return VisionModelWrapper
    elif name in nlp_models:
        if task in ["classification", "cls"]:
            return NLPClassificationModelWrapper
        elif task in ["language_modeling", "lm"]:
            return NLPLanguageModelingModelWrapper
        elif task in ["translation", "tran"]:
            return NLPTranslationModelWrapper
        else:
            raise NotImplementedError(f"Task {task} not implemented for NLP models")
    else:
        raise ValueError(f"Model {name} not implemented")
```

This is because different tasks may have a different `forward` pass and also might use different manipulations on data and also its evaluation metrics.
These `Wrapper` also defines the detailed `training_flow`, if we use `machop/chop/plt_wrapper/base.py` as an example, `training_step` and `valdiation_step` are defined in this class.

```bash

class WrapperBase(pl.LightningModule):
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        acc = self.acc_train(y_hat, y)

        self.log(
            "train_acc", self.acc_train, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
				...
```

## Models

Models are mainly implemented in `machop/chop/models/__init__.py
`, and the `model_map` contains all currently supported models.
One can also instantiate a custom model and add it to our flow like `machop/chop/models/manual/toy_manual.py`:

```python
class ToyManualNet(ManualBase):
    def __init__(self, image_size, num_classes, config=None):
        super(ToyManualNet, self).__init__(config)

        in_planes = image_size[0] * image_size[1] * image_size[2]

        linear1_config = self.config.get("linear1", None)
        linear2_config = self.config.get("linear2", None)
        linear3_config = self.config.get("linear3", None)
        if any([x is None for x in [linear1_config, linear2_config, linear3_config]]):
            raise ValueError(
                "linear1, linear2, linear3 should not be specified in config"
            )

        self.linear = nn.Sequential(
            LinearInteger(in_planes, 100, config=linear1_config),
            nn.ReLU(),
            LinearInteger(100, 100, config=linear2_config),
            nn.ReLU(),
            LinearInteger(100, num_classes, config=linear3_config),
        )

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))
```

In this case, a custom model is instantiated using the `LinearInteger` MASE-enhanced custom module.

# Test command

To test the model trained above you can use:

```bash
# After training, you will have your checkpoint under mase-tools/mase_output
# For example, the checkpoint is under mase-tools/mase_output/toy_classification_toy-tiny_2023-07-03/software/training_ckpts/best.ckpt 
./ch test --config configs/archive/test/train.toml --load ../mase_output/toy_classification_toy-tiny_2023-07-03/software/training_ckpts/best.ckpt --load-type pl
```
