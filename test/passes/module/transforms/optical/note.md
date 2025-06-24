### Note on using custom kernel for MORR linear layer

Current optical transform pass only support MORR linear PyTorch module. To enbale substitution using Optimised MORR linear module (using Triton kernel):

1. uncomment `TritonMemMORRLinear` inside [file](../../../../../src/chop/nn/optical/modules/__init__.py)
2. replace `morr_linear_fn_mem` function in [kernel wrapper](../../../../../src/chop/nn/optical/triton_modules/morr_linear_mem.py). Current implementation import it from a project file, import it from mase-triton instead.
3. You should now able to use optimised MORR linear module in optical transform pass. Two sample usage are shown below:

```python

# Minimal example ─ apply the MORR-Triton replacement to a single layer
model = Net()
pass_args = {
    "by": "name",
    "fc1": {
        "config": {
            "name": "morr_triton",
            "miniblock": 4,
            "morr_init": True,
            "trainable_morr_bias": False,
            "trainable_morr_scale": False,
        }
    },
}
new_model, _ = optical_module_transform_pass(model, pass_args)

# Use additional config to initialise MORR linear module with noise modelling
model = Net()
pass_args = {
    "by": "regex_name",
    "^fc1$": {
        "config": {"name": "morr_triton", "miniblock": 4},
        "additional": {
            "trainable_morr_bias": False,
            "trainable_morr_scale": False,
            "thermal_crosstalk": True,
            "coupling_factor": 0.04,
            "drop_perc": 0.0,
            "phase_noise": True,
            "phase_noise_std": 0.04,
            "in_bit": 8,
            "w_bit": 8,
        },
    },
}
new_model, _ = optical_module_transform_pass(model, pass_args)
```