# Zero-Cost Proxy for NAS Project with MASE

This tutorial shows how to search for neural architectures with zero-cost proxies on CIFAR-10, CIFAR-100 and ImageNet-16-120 datasets using the MASE framework.

> **Note**: The Zero-Cost Proxy project is done by Group 2 of Advanced deep learning system course at Impeiral college london.

## Search for Neural Architectures with Zero-Cost Proxies

We load the NAS-Bench-201 benchmark API and use the MASE framework to search for neural architectures with zero-cost proxies. Architectures with its true post-train accuracy are selected from NAS-Bench-201 and searched with zero-cost proxies as objectives.

### Requirements

Before initiating the project, please ensure the following prerequisites are satisfied:

- Python (same with MASE)
- PyTorch (same with MASE)
- NAS-Bench-201 API
- An environment capable of running PyTorch models, such as CUDA for GPU acceleration

### Sample search Config

Here is the search part in `configs/examples/search_zero_cost.toml` looks like the follows.

```toml
[search.search_space]
# the search space named defined in mase
# should be set as "graph/software/zero_cost" for zero-cost search
name = "graph/software/zero_cost"

[search.search_space.setup]
by = "name"

[search.search_space.seed.default.config]
# the only choice "NA" is used to indicate that layers are not quantized by default
name = ["NA"]

### search config for vision
[search.search_space.nas_zero_cost.config]
## default cifar 10 (can be selected from [cifar10, cifar10-valid, cifar100, ImageNet16-120])
dataset = ['ImageNet16-120'] 
name = ['infer.tiny']
C = [16]
N = [5]

## following are the option choices for each node in the cell, 
## each of them can be chosen from 0,1,2,3,4
op_1_0 = [0,1,2]
op_2_0 = [2,3,4]
op_2_1 = [0, 4]
op_3_0 = [4]
op_3_1 = [0,1]
op_3_2 = [1,2]

number_classes = [10]

[search.strategy]
## use optuna as the optimization algorithm
name = "optuna"
## should be set to false because zero-cost nas requires a mini-batch of training
eval_mode = false

[search.sw_runner]
## set to "zero_cost" to call the newly-defined zc runner that can return both the zero-cost proxies and the true accuracy values
name = "zero_cost"

[search.strategy.sw_runner.zero_cost]
# metric can be chosen from 
# "grad_norm", "snip", "grasp", "fisher", "plain", "l2_norm", "naswot", "naswot_relu", "tenas", "zico"
# choose three of zero cost proxies
metrics = ["naswot_relu", "snip", "plain"]
data_loader = "val_dataloader"
num_samples = 512

[search.strategy.setup]
n_jobs = 1

# change the number of trail here
n_trials = 5

timeout = 20000
sampler = "tpe"
# sum_scaled_metrics should be false for multi-objective optimization
sum_scaled_metrics = false # multi objective
# direction needs to be commented out for multi-objective optimization
# direction = "maximize"

[search.strategy.metrics]
### set the scale for the chosen zero cost proxy
# grad_norm.scale = 0.0
# grad_norm.direction = "maximize"
snip.scale = 1.0
snip.direction = "maximize"
# grasp.scale = 0.0
# grasp.direction = "maximize"
# fisher.scale = 0.0
# fisher.direction = "maximize"
# jacob_cov.scale = 1.0
# jacob_cov.direction = "maximize"
plain.scale = 1.0
plain.direction = "maximize"
# synflow.scale = 1.0
# synflow.direction = "maximize"
# l2_norm.scale = 0.0
# l2_norm.direction = "minimize"
# naswot.scale = 1.0
# naswot.direction = "maximize"
naswot_relu.scale = 1.0
naswot_relu.direction = "maximize"
# t_cet.scale = 1.0
# t_cet.direction = "maximize"
# tenas.scale = 1.0
# tenas.direction = "maximize"
# zen.scale = 1.0
# zen.direction = "maximize"
# zico.scale = 0.0
# zico.direction = "maximize"
```

### NAS-Bench-201 Dataset Requirement

First download the NAS-Bench-201 `.pth` file. This file contains the dataset of pre-evaluated architectures for the zero-cost proxy evaluation process.

Download the file from the official NAS-Bench-201 repository or an alternative provided source. After downloading, place the `.pth` file in your project directory under `mase/machop/third-party/NAS-Bench-201-v1_1-096897.pth`.

#### Setup Instructions

1. **Create and Activate a Python Virtual Environment**:

   ```bash
   conda env create -f machop/environment.yml
   conda activate mase
   pip install -r machop/requirements.txt
   ```
2. **Install Required Dependencies**:

   ```bash
   pip install torch pandas optuna nas-bench-201-api
   ```

   Then uncomment line 30, 53-55 of file `/mase/machop/chop/actions/search/search_space/zero_cost/graph.py`.
3. Next, install the nas-bench-201 pth file to this path: `mase/machop/third-party/` using the following link:
   [Google Drive NAS-Bench](https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view?usp=sharing)
4. **Configuration Check**: Ensure that the `configs/examples/search_zero_cost.toml` configuration file points to the correct NAS-Bench-201 `.pth` file location and adjust other settings if necessary.

### Project Execution

Execute the project by following command under `mase/machop` directory:

```bash
./ch search --config configs/examples/search_zero_cost.toml
```

This command triggers the search procedure that employs zero-cost proxies to evaluate and rank neural network architectures. The process includes the model and dataset initialization, search space construction, and the execution of the zero-cost proxy search strategy, followed by the logging and saving of results.

## Expected Outputs

Upon successful completion, the project generates:

- Predicted accuracy rankings of neural network architectures.
- True accuracy values as evaluated on the NAS-Bench-201 dataset.
- Log files and visualizations, depending on your setup and configurations.

These outputs are crucial for assessing the performance of various architectures and the predictive accuracy of zero-cost proxies.

For instance, given the aforementioned configuration:

After 5 trails is done: the terminal will show the following information.

```python
INFO     Building search space...
INFO     Search started...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [08:09<00:00, 97.93s/it, 489.66/20000 seconds]
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                                          | hardware_metrics   | scaled_metrics                                            |
|----+----------+-----------------------------------------------------------+--------------------+-----------------------------------------------------------|
|  0 |        3 | {'naswot_relu': 402.299, 'snip': 127.374, 'plain': 0.175} | {}                 | {'naswot_relu': 402.299, 'plain': 0.175, 'snip': 127.374} |
INFO     Searching is completed
```

In addition, two extra `.json` files called `zc_ensemble.json` and `zc_with_predivtive_and_true.json` will be automatically saved to `mase_output/[project_name]/sofware/search_ckpts`. 

`zc_ensemble.json` will save the ensemble weights for each of the selected zero cost proxies, showed as follows.
```json
{
    "weights":{
        "intercept":41.4959999701,
        "naswot_relu":-1.1211446418,
        "plain":3.040506729,
        "snip":3.4640618783
    }
}
```

`zc_with_predivtive_and_true.json` will save the standardised zero cost proxies for each trail, the true accuracy retrived from the NAS-Bench-201, and the predictive accuracy by the regression model. A sample file is shown as below.

```json
{
    "0":{
        "naswot_relu":-0.7256577606,
        "plain":0.6161797275,
        "snip":0.416254574,
        "Predicted_Accuracy":45.6249974894,
        "True_Accuracy":45.679999939
    },
    "1":{
        "naswot_relu":-1.1341014803,
        "plain":-1.1690386788,
        "snip":0.4822234451,
        "Predicted_Accuracy":40.8834736517,
        "True_Accuracy":40.84
    },
    "2":{
        "naswot_relu":0.1737019535,
        "plain":-0.4977747846,
        "snip":-1.100214481,
        "Predicted_Accuracy":35.9765563322,
        "True_Accuracy":36.9399999908
    },
    "3":{
        "naswot_relu":1.4495217146,
        "plain":1.3779527079,
        "snip":1.1879220416,
        "Predicted_Accuracy":48.1755864059,
        "True_Accuracy":48.2199999756
    },
    "4":{
        "naswot_relu":0.2365355728,
        "plain":-0.327318972,
        "snip":-0.9861855797,
        "Predicted_Accuracy":36.8193859714,
        "True_Accuracy":35.7999999451
    }
}
```

## Conclusion

We finished all basic elements and one of the extensions for the project assignments.

- [X]  Design a search space for zero-cost proxy search using **option choices** for each node in the cell that construct the architecture rather than solely index.
- [X]  Implement a zero-cost proxy search strategy using TPE.
- [X]  Evaluate the performance of zero-cost proxies on CIFAR-10.
- [X]  Estimate an ensemble for the proxy combination using linear regression.
- [X]  Broaden the search to larger datasets such as CIFAR-100 and ImageNet-16-120.

<!-- The explanation documentation is under the `machop/sphinx_docs/source/modules/documentation/tutorials/actions/search` directory. -->
