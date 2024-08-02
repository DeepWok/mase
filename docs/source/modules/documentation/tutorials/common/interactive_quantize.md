# Importing MASE Quantizers 

An easy way to use MASE is to make a direct import.

As mentioned in Getting Started, you can install an editable version of MASE through pip by

```sh
pip install -e . -vvv
```

You can then test the installation by 

```sh
python -c"import chop; print(chop)"
```

## Importing MASE Internal Quantizers

MASE has a great number of custom quantizers that can be used to quantize a model.
Users can have access to these quantizers by importing them directly, without invoking any MASE passes.

For instance, one usage is to directly use the `quantizers` that we have implemented in MASE.

```python
from chop.nn.quantizers import quantizer_map

# Get the quantizer
quantizer = quantizer_map['integer']
# Apply the quantizer to a tensor
x = torch.randn(10, 10)
xq = quantizer(x, width=8, frac_width=4)
print(xq)
```

## Importing MASE Quantized Modules or Functions

Another usage is to directly use the quantized modules or functions that we have implemented in MASE.

```python 
import torch
from chop.nn.quantized.functional import quantized_func_map
from chop.nn.quantized.modules import quantized_module_map

# this shows you the possible quantized functions and modules that you can use.
print(quantized_func_map.keys())
print(quantized_module_map.keys())

add_fn = quantized_func_map['add_integer']

# Apply the quantized function to two tensors
x1 = torch.randn(10, 10)
x2 = torch.randn(10, 10)
config = {
    "data_in_width": 8,
    "data_in_frac_width": 4,
}
y = add_fn(x1, x2, config=config)


# Apply the quantized module
linear_module = quantized_module_map['linear_integer']
quantize_config = {
    "weight_width": 8,
    "weight_frac_width": 4,
    "data_in_width": 8,
    "data_in_frac_width": 4,
    "data_out_width": 8,
    "data_out_frac_width": 4,
    "bias_width": 8,
    "bias_frac_width": 4,
}
linear = linear_module(in_features=10, out_features=10, config=quantize_config)
x = torch.randn(10, 10)
y = linear(x)
```