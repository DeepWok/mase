# Run the transform action with the CLI

## Train a model

```bash
./ch train toy toy_tiny --config ../configs/archive/test/train.toml --max-epochs 3
```

## Quantise transform

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

## Quantization-aware Training (QAT)

- Load and do Quantisation-aware-training (QAT) with a transformed model

```bash
./ch train --config ../configs/examples/toy_uniform.toml --model toy --load your_post_transform_mz --load-type mz  --max-epochs 3
```

- Quantise it with fixed-point quantisers
- The load path `--load` changes with your generation time of course

> **_NOTE:_** `transform` by default saves transformed models in a mase format called `xxx.mz`, fine-tuning the transformed model only need to load this `.mz` format.

## Quantise transform by Type

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
