# Example: Mixed-precision search

This tutorial shows how to search for mixed-precision quantization strategy for Toy model on ToyTiny dataset.

## Train a Toy model on ToyTiny dataset

First we train a Toy model on ToyTiny dataset. After training for 20 epochs, we get a Toy model with around 0.997 validation accuracy. The checkpoint is saved at `mase_tools/toy_on_toy_tiny/software/training_ckpts/best.ckpt`


```bash
cd machop
./ch search --config configs/examples/search_toy_tpe.toml
```

## Search for mixed-precision quantization strategy

We load the trained Toy and search for fixed-point precision.

### Search Config

Here is the search part in `configs/examples/search_toy_tpe.toml` looks like the following.

```toml
[search.search_space]
# the search space name defined in mase
# this `name="graph/quantize/mixed_precision_ptq"` will create a mixed-precision post-training-quantization search space
name = "graph/quantize/mixed_precision_ptq"

[search.search_space.setup]
# the config for MixedPrecisionSearchSpace
# this `by="name"` will quantize the model by node/layer name when rebuilding the model
by = "name"

[search.search_space.seed.default.config]
# the default quantization config the node/layer
name = ["integer"]
data_in_width = [4, 8]
data_in_frac_width = [3, 4, 5, 6, 7, 8, 9]
weight_width = [2, 4, 8]
weight_frac_width = [3, 4, 5, 6, 7, 8, 9]
bias_width = [2, 4, 8]
bias_frac_width = [3, 4, 5, 6, 7, 8, 9]

[search.strategy]
# the search strategy name "optuna" specifies the search algorithm
name = "optuna"
sw_runner = "basic_evaluation" # sw_runner specifies the estimator for sw metrics
hw_runner = "average_bitwidth" # hw_runner specifies the estimator for hw metrics
eval_mode = true # set the model in eval mode since we are doing post-training quantization
data_loader = "val_dataloader"
num_samples = 256

[search.strategy.setup]
# the config for SearchStrategyOptuna
n_jobs = 1
n_trials = 20
timeout = 20000
sampler = "tpe"
# sum_scaled_metrics = true # single objective
# direction = "maximize"
sum_scaled_metrics = false # multi objective

[search.strategy.metrics]
# the sw + hw metrics to be evaluated
# since the loss is commented out, the search objective will not include this term
# loss.scale = 0.0
# loss.direction = "minimize"
accuracy.scale = 1.0
accuracy.direction = "maximize"
average_bitwidth.scale = 0.2
average_bitwidth.direction = "minimize"
```

### Run the search

Run the following command to start the search. We search for 20 trials and save the results in `mase_tools/toy_toy_tiny/software/search_results`.

```bash
./ch search --config configs/examples/search_toy_tpe.toml
```

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