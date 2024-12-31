# Mixed-precision search on MASE Graph

This tutorial shows how to search for mixed-precision quantization strategy for JSC model (a small toy model).

## Commands

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




## Search Config

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

## Run the search

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
