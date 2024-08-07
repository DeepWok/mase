# Mixed-precision search on manual model

This tutorial shows how to search for mixed-precision quantization strategy for OPT model on Wikitext2 dataset.

> **Note**: Manual model refers to the model named as `<model_arch>_quantized` at `mase-tools/machop/chop/models/manual`. Usually these are models that cannot be directly converted to MASE Graph.

## Search for Mixed-Precision Quantization Scheme

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

### Launch the Precision Search

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

### Search Logs

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