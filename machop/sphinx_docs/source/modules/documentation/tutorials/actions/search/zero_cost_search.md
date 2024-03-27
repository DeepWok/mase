# Zero Cost Proxy Network Architecture Search (NAS)

This tutorial shows how to do a zero cost proxy NAS search using different datasets and benchmarks.

## Search Config
```toml
# the search space name defined in mase
# this `name="graph/zero_cost_proxy"` will create a zero cost search space
[search.search_space]
name = "graph/zero_cost_proxy"

# configuration settings for doing a zero cost NAS
[search.search_space.zc]
seed = 2
benchmark = 'nasbench201' # the benchmark you are using, one of [nasbench201]
dataset = 'cifar10' # the dataset you are using, one of ['cifar10', 'cifar100', 'ImageNet16-120']
calculate_proxy = false # whether to calculate the proxy from scratch or look them up from an api
ensemble_model = 'nonlinear' # for the neural network ensemble model, one of ['linear', 'nonlinear']
loss_fn = 'mae' # the loss function to use for the 'ensemble_model', one of ['mse', 'mae', 'huber']
optimizer = 'adam' # the optimizer to use for the 'ensemble_model', one of ['adam', 'adamW', 'rmsProp']
batch_size = 4 # the batch size to use for the 'ensemble_model'
learning_rate = 0.02 # the learning rate for the 'ensemble_model'
epochs = 30 # the number of epochs to train the 'ensemble_model' for
num_archs_train = 2000 # the number of architectures to use to train 'ensemble_model'
num_archs_test = 2000 # the number of architectures to use to test 'ensemble_model'
zc_proxies = ['epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'plain', 'snip', 'synflow', 'zen', 'flops', 'params'] # the zero cost proxies to evaluate

# the search strategy name "zero_cost" specifies the search algorithm
[search.strategy]
name = "zero_cost"

[search.strategy.sw_runner.basic_evaluation]
data_loader = "val_dataloader"
num_samples = 512

[search.strategy.hw_runner.average_bitwidth]
compare_to = 32

# the config for SearchStrategyZeroCost
[search.strategy.setup]
n_jobs = 4
n_trials = 100
timeout = 20000
sampler = "tpe"
direction = "minimize"
weight_lower_limit = 0
weight_upper_limit = 30
```

## Run the search
Run the following command to start the search. We search for 100 trials and save the results in `mase_output/zero_cost_proxy_test/software/search_ckpts`.

```bash
./ch search --config configs/examples/zero_cost_proxy.toml
```

When the search is completed, you will see the best metrics printed in the terminal, along with the parameters used. 
The 'spearman' column ranks the top 5 best metrics based on the Spearman correlation, the 'kendaltau' column ranks the top 5 best metrics based on the Kendal tau correlation, and the 'Global Parameters' column shows 5 parameters set in the search configuration file. 

```text
|    | spearman                          | kendaltau                         | Global Parameters            |
|----+-----------------------------------+-----------------------------------+------------------------------|
|  0 | {'xgboost': 0.928}                | {'xgboost': 0.789}                | {'num_training_archs': 2000} |
|  1 | {'nonlinear': 0.82}               | {'nonlinear': 0.632}              | {'num_testing_archs': 2000}  |
|  2 | {'optuna_ensemble_metric': 0.779} | {'optuna_ensemble_metric': 0.594} | {'dataset': 'cifar10'}       |
|  3 | {'synflow': 0.774}                | {'synflow': 0.581}                | {'benchmark': 'nasbench201'} |
|  4 | {'nwot': 0.756}                   | {'nwot': 0.57}                    | {'num_zc_proxies': 13}       |
```

The entire searching log is saved in `mase_output/zero_cost_proxy_test/software/search_ckpts/log.json`.

Here is part of the `log.json`:

```json
{
    "0":{
        "number":0,
        "value":3.003744989e+18,
        "state":"COMPLETE",
        "datetime_start":1710681883223,
        "datetime_complete":1710681883235,
        "duration":11
    },
    "1":{
        "number":1,
        "value":8.584398984e+17,
        "state":"COMPLETE",
        "datetime_start":1710681883236,
        "datetime_complete":1710681883244,
        "duration":8
    },
    "2":{
        "number":2,
        "value":2.161143026e+17,
        "state":"COMPLETE",
        "datetime_start":1710681883245,
        "datetime_complete":1710681883254,
        "duration":9
    }
}
```

Additionally, the metrics is saved in `mase_output/zero_cost_proxy_test/software/search_ckpts/metrics.json`.

This file contains the results of each test architecture which includes the zero cost metric for each zero cost proxy, as well as the ensemble weight assigned to that architecture when using Optuna. 

Additionally, the file contains the Spearman and Kendal tau correlations for all the single zero cost proxies, as well as the results of the three ensemble methods; 
1. Optuna ensemble
2. linear/nonlinear neural network ensemble
3. XGBoost ensemble

Here is part of the `metrics.json`:

```
{
    "0":{
        "number":0,
        "result_dict":{
            "optuna_ensemble_metric":{
                "test_spearman":0.7787709203,
                "test_kendaltau":0.5939566279
            },
            "nonlinear":{
                "test_spearman":0.8197650448,
                "test_kendaltau":0.6324463152
            },
            "xgboost":{
                "test_spearman":0.927857595,
                "test_kendaltau":0.7888095736
            },
            "epe_nas":{
                "test_spearman":0.6127902671,
                "train_spearman":0.6321602372,
                "test_kendaltau":0.4533952783
            },
            "fisher":{
                "test_spearman":0.5315838306,
                "train_spearman":0.4977034213,
                "test_kendaltau":0.378125962
            },
            "grad_norm":{
                "test_spearman":0.6226992615,
                "train_spearman":0.5880364388,
                "test_kendaltau":0.4490600814
            },
            "grasp":{
                "test_spearman":0.5893576198,
                "train_spearman":0.5664100565,
                "test_kendaltau":0.4056164133
            },
            "jacov":{
                "test_spearman":0.6505983247,
                "train_spearman":0.668493689,
                "test_kendaltau":0.4928354368
            },
            "l2_norm":{
                "test_spearman":0.6653796055,
                "train_spearman":0.6523610029,
                "test_kendaltau":0.4779827799
            },
            "nwot":{
                "test_spearman":0.7563704105,
                "train_spearman":0.7432270476,
                "test_kendaltau":0.5695074915
            },
            "plain":{
                "test_spearman":-0.1957988224,
                "train_spearman":-0.2107663188,
                "test_kendaltau":-0.1337404731
            },
            "snip":{
                "test_spearman":0.6315350754,
                "train_spearman":0.5984573,
                "test_kendaltau":0.4581087138
            },
            "synflow":{
                "test_spearman":0.7736406307,
                "train_spearman":0.7585138544,
                "test_kendaltau":0.5805102044
            },
            "zen":{
                "test_spearman":0.375039313,
                "train_spearman":0.3605840113,
                "test_kendaltau":0.2977154582
            },
            "flops":{
                "test_spearman":0.7102408238,
                "train_spearman":0.6731944804,
                "test_kendaltau":0.5181277058
            },
            "params":{
                "test_spearman":0.7227988829,
                "train_spearman":0.6920328711,
                "test_kendaltau":0.5453489131
            }
        },
        "model_results":[
            {
                "test_hash":"(2, 1, 2, 2, 1, 1)",
                "test_accuracy":99.868,
                "metrics":{
                    "epe_nas":1778.510433327,
                    "fisher":0.0969865993,
                    "grad_norm":24.9073505402,
                    "grasp":2.8195419312,
                    "jacov":-65.245303883,
                    "l2_norm":180.8004150391,
                    "nwot":786.86604851,
                    "plain":0.026368428,
                    "snip":41.5649223328,
                    "synflow":47.4053850614,
                    "zen":82.5358505249,
                    "flops":115.402432,
                    "params":0.802426,
                    "optuna_ensemble":5870.7588437487
                }
            },
            ...
        ]
    }
}
```
