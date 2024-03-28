# Zero-Cost Proxy for NAS Project with MASE

This tutorial shows how to search for neural architectures with zero-cost proxies on CIFAR-10 and ImageNet-16-120 datasets using the MASE framework.

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

## following are the option choices for each node in the cell
op_0_0 = [0]
op_1_0 = [0,1,2,3,4]
op_2_0 = [0,1,2,3,4]
op_2_1 = [0,1,2,3,4]
op_3_0 = [0,1,2,3,4]
op_3_1 = [0,1,2,3,4]
op_3_2 = [0,1,2,3,4]

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
metrics = ["grad_norm", "snip", "grasp", "fisher", "plain", "l2_norm", "naswot", "naswot_relu", "tenas", "zico"]
data_loader = "val_dataloader"
num_samples = 512

[search.strategy.hw_runner.average_bitwidth]
compare_to = 32 # compare to FP32

[search.strategy.setup]
n_jobs = 1
n_trials = 100
timeout = 20000
sampler = "tpe"
# sum_scaled_metrics should be false for multi-objective optimization
sum_scaled_metrics = false # multi objective
# direction needs to be commented out for multi-objective optimization
# direction = "maximize"

[search.strategy.metrics]
grad_norm.scale = 0.0
grad_norm.direction = "maximize"
snip.scale = 1.0
snip.direction = "maximize"
grasp.scale = 0.0
grasp.direction = "maximize"
fisher.scale = 0.0
fisher.direction = "maximize"
plain.scale = 1.0
plain.direction = "maximize"
l2_norm.scale = 0.0
l2_norm.direction = "minimize"
naswot.scale = 1.0
naswot.direction = "maximize"
naswot_relu.scale = 1.0 # number 3
naswot_relu.direction = "maximize"
t_cet.scale = 1.0
t_cet.direction = "maximize"
tenas.scale = 1.0
tenas.direction = "maximize"
zen.scale = 1.0
zen.direction = "maximize"
zico.scale = 0.0
zico.direction = "maximize"
```

### NAS-Bench-201 Dataset Requirement

First download the NAS-Bench-201 `.pth` file. This file contains the dataset of pre-evaluated architectures for the zero-cost proxy evaluation process.

Download the file from the official NAS-Bench-201 repository or an alternative provided source. After downloading, place the `.pth` file in your project directory under `third_party/NAS-Bench-201-v1_1-096897.pth` or adjust the configuration to reflect the file's location accurately.

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

   Then uncomment line 47, 76-78 of file /zero_cost/graph.py.
3. Next, install the nas-bench-201 pth file to this path:'./third_party/' using the following link:
   [Google Drive NAS-Bench](https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view?usp=sharing)
4. **Configuration Check**: Ensure that the `configs/examples/search_zero_cost.toml` configuration file points to the correct NAS-Bench-201 `.pth` file location and adjust other settings as necessary.

### Project Execution

Execute the project by following command from the root directory:

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

For instance, given the following configuration:

```toml
op_0_0 = [0]
op_1_0 = [0,1,2,3,4]
op_2_0 = [0,1,2,3,4]
op_2_1 = [0,1,2,3,4]
op_3_0 = [0,1,2,3,4]
op_3_1 = [0,1,2,3,4]
op_3_2 = [0,1,2,3,4]
```

After a complete search run:

```python
INFO    Best trial(s):
|    |   number | software_metrics                                            | scaled_metrics                                              | nasbench_data_metrics                                                                                                                                                                            |
|----+----------+-------------------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 |        9 | {'snip': 301.718, 'plain': -0.025, 'naswot_relu': 6513.572} | {'naswot_relu': 6513.572, 'plain': -0.025, 'snip': 301.718} | {'train-loss': 0.362, 'train-accuracy': 87.7, 'train-per-time': 32.842, 'train-all-time': 394.101, 'test-loss': 0.443, 'test-accuracy': 85.11, 'test-per-time': 1.93, 'test-all-time': 23.163}   |
|  1 |       32 | {'snip': 31.569, 'plain': -0.017, 'naswot_relu': 6539.923}  | {'naswot_relu': 6539.923, 'plain': -0.017, 'snip': 31.569}  | {'train-loss': 0.221, 'train-accuracy': 92.494, 'train-per-time': 28.797, 'train-all-time': 345.57, 'test-loss': 0.344, 'test-accuracy': 88.79, 'test-per-time': 1.818, 'test-all-time': 21.821} |
|  2 |       64 | {'snip': 425.625, 'plain': 0.157, 'naswot_relu': 6376.482}  | {'naswot_relu': 6376.482, 'plain': 0.157, 'snip': 425.625}  | {'train-loss': 0.673, 'train-accuracy': 76.336, 'train-per-time': 27.916, 'train-all-time': 334.993, 'test-loss': 0.723, 'test-accuracy': 74.53, 'test-per-time': 1.73, 'test-all-time': 20.766} |
...
```

In addition, a weights series containing estimated importance weights to each proxy will also return.

```python
[7.48018255e+00 -2.42037272e-01 -3.68762718e+00]
```

## Conclusion

We finished all basic elements and one of the extensions for the project assignments.

- [X]  Design a search space for zero-cost proxy search using **option choices** for each node in the cell that construct the architecture rather than solely index.
- [X]  Implement a zero-cost proxy search strategy using TPE.
- [X]  Evaluate the performance of zero-cost proxies on CIFAR-10.
- [X]  Estimate an ensemble for the proxy combination using linear regression.
- [X]  Broaden the search to larger datasets such as CIFAR-100 and ImageNet-16-120.

<!-- The explanation documentation is under the `machop/sphinx_docs/source/modules/documentation/tutorials/actions/search` directory. -->
