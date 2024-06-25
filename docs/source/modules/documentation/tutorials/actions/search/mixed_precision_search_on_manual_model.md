# Mixed-precision search on Manual model

This tutorial shows how to search for mixed-precision quantization strategy for OPT model on Wikitext2 dataset.

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

[search.strategy.sw_runner.basic_evaluation]
data_loader = "val_dataloader"
num_samples = 512

[search.strategy.hw_runner.average_bitwidth]
compare_to = 32 # compare to FP32

[search.strategy.setup]
# Optuna supports a range of search algorithms, including Random, TPE, Genetic, etc.
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
average_bitwidth.scale = 1.0
average_bitwidth.direction = "minimize"
```

Run the search:
```bash
cd machop
./ch search --config ../configs/examples/search_opt_quantized_tpe_search.toml
```

When the search is done, the best quantization config will be printed out:
```txt
Best trial(s):
|    |   number | software_metrics                        | hardware_metrics                                     | scaled_metrics |
|----+----------+-----------------------------------------+------------------------------------------------------+----------------|
|  0 |        4 | {'loss': 8.633, 'perplexity': 1011.627} | {'average_bitwidth': 5.194, 'memory_density': 6.16}  | ...            |
|  1 |        5 | {'loss': 8.578, 'perplexity': 1088.06}  | {'average_bitwidth': 5.139, 'memory_density': 6.227} | ...            |
|  2 |        6 | {'loss': 9.341, 'perplexity': 409.964}  | {'average_bitwidth': 5.958, 'memory_density': 5.371} | ...            |
|  3 |        8 | {'loss': 9.527, 'perplexity': 191.928}  | {'average_bitwidth': 6.514, 'memory_density': 4.913} | ...            |
|  4 |        9 | {'loss': 9.426, 'perplexity': 214.558}  | {'average_bitwidth': 6.333, 'memory_density': 5.053} | ...            |
```

Usually the TPE can optimize the average bitwidth and perplexity trade-off.

The complete search results will be saved in `mase/mase_output/opt_quantized_wikitext2/software/search_ckpts/log.json`.

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