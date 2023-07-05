
# Train a model

```bash
./ch train --config configs/archive/test/train.toml --model toy
```

# Quantise transform

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
./ch transform --config configs/examples/toy_uniform.toml --model toy --load ../mase_output/toy_classification_toy-tiny_2023-07-05/software/training_ckpts/best.ckpt --load-type pl
```

# QAT

- Load and do Quantisation-aware-training (QAT) with a transformed model

```bash
./ch train --config configs/examples/toy_uniform.toml --model toy --load ../mase_output/toy_classification_
toy-tiny_2023-07-05/software/transformed_ckpts/graph_module.mz --load-type mz
```


- Quantise it with fixed-point quantisers
- The load path `--load` changes with your generation time of course

# Quantise transform by type

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

In this case, only linear layers are quantised!

```bash
./ch transform --config configs/examples/toy_by_type.toml --model toy --load ../mase_output/toy_classification_toy-tiny_2023-07-05/software/training_ckpts/best.ckpt --load-type pl
```

# Direct training of a quantized model