# MaseModule Passes

`Modules` in `torch` can be seen as composable building blocks for neural networks. They are defined by subclassing `torch.nn.Module` and implementing the `forward` method. More detail on `torch.nn.Module` can be found [here](https://pytorch.org/docs/stable/notes/modules.html).

In `mase`, we support directly perform passes on `torch.nn.Module` objects.

```python
import torch
from chop.passes.module.transforms import quantize_module_transform_pass

class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 28 * 28)
        self.fc2 = nn.Linear(28 * 28, 28 * 28 * 4)
        self.fc3 = nn.Linear(28 * 28 * 4, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        # w = torch.randn((4, 28 * 28))
        # x = torch.nn.functional.relu(nn.functional.linear(x, w))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


mlp = MLP()
# Sanity check and report
pass_args = {
		"by": "name",
		"fc1": {
				"name": "integer",
				"data_in_width": 8,
				"data_in_frac_width": 4,
				"weight_width": 8,
				"weight_frac_width": 4,
				"bias_width": 8,
				"bias_frac_width": 4,
		},
}
# directly apply quantization on top of a native torch model
quantize_module_transform_pass(mlp, pass_args)
```