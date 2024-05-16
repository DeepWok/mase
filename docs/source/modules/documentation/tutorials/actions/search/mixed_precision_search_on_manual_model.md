# Mixed-precision search on Manual model

This tutorial shows how to search for mixed-precision quantization strategy for OPT_quantized model on Wikitext2 dataset.

> **Note**: Manual model refers to the model named as `<model_arch>_quantized` at `mase-tools/machop/chop/models/manual`. Usually these are models that cannot be directly converted to MASE Graph.

## Search for mixed-precision quantization strategy

We load the HuggingFace checkpoint "facebook/opt-125m" and search for fixed-point precision on Wikitext2 dataset.

### Search config

Here is the search part in `configs/examples/search_opt_quantized_tpe_search.toml` looks like the following.

```toml
[search.search_space]
# the search space name defined in mase
name = "module/manual_hf/quantize/llm_mixed_precision_ptq"

[search.search_space.setup]
# disable model parallel since we are using a small model
model_parallel = false

[search.search_space.seed.default]
# Since we are doing mixed-precision search.
# Only one "name" is supported (len(name) == 1)
name = ["integer"]
data_in_width = [2, 4, 8, 10]
data_in_frac_width = [2, 4, 6]
weight_width = [2, 4, 8, 10]
weight_frac_width = [2, 4, 6]
bias_width = [2, 4, 8, 10]
bias_frac_width = [2, 4, 6]

[search.strategy]
# Optuna supports a range of search algorithms, including Random, TPE, Genetic, etc.
name = "optuna"
sw_runner = "basic_evaluation" # sw_runner specifies the estimator for sw metrics
hw_runner = "average_bitwidth" # hw_runner specifies the estimator for hw metrics
eval_mode = true
data_loader = "val_dataloader"
num_samples = 16

[search.strategy.setup]
n_jobs = 1
n_trials = 10
timeout = 20000
sampler = "TPE"
model_parallel = false
runner_style = "lm"
sum_scaled_metrics = false

[search.strategy.metrics]
perplexity.scale = 1.0
perplexity.direction = "minimize"
# we set `average_bitwidth.scale = 0.0` to disable the hw metric,
# since currently (commit bb44d2cf255fae54ecb6697b0fe23c595197babf) the bitwidth estimation only profiles linear layers
average_bitwidth.scale = 0.0
average_bitwidth.direction = "minimize"
```

Run the search:
```bash
cd machop
./ch search --config configs/examples/search_opt_quantized_tpe_search.toml
```

When the search is done, the best quantization config will be printed out:
```txt
Best trial(s):
|    |   number | software_metrics                     | hardware_metrics            | scaled_metrics                                   |
|----+----------+--------------------------------------+-----------------------------+--------------------------------------------------|
|  0 |        9 | {'loss': 9.6, 'perplexity': 144.111} | {'average_bitwidth': 5.556} | {'average_bitwidth': 0.0, 'perplexity': 144.111} |
```

The complete search results will be saved in `mase_tools/mase_output/opt_quantized_wikitext2/software/search_ckpts/log.json`.

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
    ...
}
```