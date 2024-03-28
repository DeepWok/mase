# Running the Zero-Cost Proxy Project

This document guides you through the process of running the Zero-Cost Proxy project, which focuses on optimizing neural network architectures utilizing zero-cost proxies. Follow the steps detailed below to prepare and execute the project efficiently.

## Requirements

Before initiating the project, please ensure the following prerequisites are satisfied:

- Python (same with MASE)
- PyTorch (same with MASE)
- Optuna
- Pandas
- NAS-Bench-201 API
- An environment capable of running PyTorch models, such as CUDA for GPU acceleration

### NAS-Bench-201 Dataset Requirement

Downloading the NAS-Bench-201 `.pth` file is a crucial step. This file contains the dataset of pre-evaluated architectures essential for the zero-cost proxy evaluation process.

Download the file from the official NAS-Bench-201 repository or an alternative provided source. After downloading, place the `.pth` file in your project directory under `third_party/NAS-Bench-201-v1_1-096897.pth` or adjust the configuration to reflect the file's location accurately.

## Setup Instructions

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

   Then install the nas-bench-201 pth file using the following link:
   [Google Drive NAS-Bench](https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view?usp=sharing)
3. **Configuration Check**: Ensure that the `configs/examples/search_zero_cost.toml` configuration file points to the correct NAS-Bench-201 `.pth` file location and adjust other settings as necessary for your project.

## Project Execution

Execute the project by running the following command from the root directory:

```bash
./ch search --config configs/examples/search_zero_cost.toml
```

This command triggers the search procedure that employs zero-cost proxies to evaluate and rank neural network architectures, utilizing data from the NAS-Bench-201 dataset. The process includes the model and dataset initialization, search space construction, and the execution of the zero-cost proxy search strategy, followed by the logging and saving of results.

## Expected Outputs

Upon successful completion, the project generates:

- Predicted accuracy rankings of neural network architectures.
- True accuracy values as evaluated on the NAS-Bench-201 dataset.
- Log files and visualizations, depending on your setup and configurations.

These outputs are crucial for assessing the performance of various architectures and the predictive accuracy of zero-cost proxies.

For instance, given the following configuration:

```python
{'nas\_zero\_cost': {'dataset': 'ImageNet16-120', 'name': 'infer.tiny', 'C': 16, 'N': 5, 'op\_0\_0': 0, 'op\_1\_0': 1, 'op\_2\_0': 2, 'op\_2\_1': 3, 'op\_3\_0': 0, 'op\_3\_1': 2, 'op\_3\_2': 4, 'number\_classes': 10}, 'default': {'config': {'dataset': 'cifar10', 'name': 'infer.tiny', 'C': 16, 'N': 5, 'op\_0\_0': 0, 'op\_1\_0': 4, 'op\_2\_0': 2, 'op\_2\_1': 1, 'op\_3\_0': 2, 'op\_3\_1': 1, 'op\_3\_2': 1, 'number\_classes': 10}}}
```

After a complete search run:

```python
INFO    Best trial(s):
|    |   number | software_metrics | hardware_metrics | scaled_metrics | nasbench_data_metrics |
|----|----------|------------------|------------------|----------------|-----------------------|
...
```

Proxy Logger and Sorted Results visualization:

```python
INFO    Proxy Logger:
    fisher  grad_norm  grasp  l2_norm    naswot  naswot_relu     plain      snip  zico
...
INFO    Sorted Results:
    Architecture_Index  Predicted_Accuracy  True Accuracy
...

```

These outputs are crucial for assessing the performance of various architectures and the predictive accuracy of zero-cost proxies.

## Conclusion

This guide facilitates the running of the Zero-Cost Proxy project for neural network architecture evaluation. This methodology offers a computationally efficient alternative to traditional architecture optimization methods, proving invaluable for machine learning researchers and practitioners.
