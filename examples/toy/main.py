# This example converts a simple MLP model to MLIR, Torch script code and ONNX graph
import torch
import torch_mlir
import torch.nn as nn

# Model specifications
class MLP(torch.nn.Module):
    # Toy quantized FC model for digit recognition on MNIST

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 28 * 28)
        self.fc2 = nn.Linear(28 * 28, 28 * 28 * 4)
        self.fc3 = nn.Linear(28 * 28 * 4, 10)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
mlp = MLP()

# Inputs to the model
batch_size = 2
x = torch.randn((batch_size, 28, 28))

# Emit the model to Linalg MLIR
# Better to print to the file - the constants are a bit long
module = torch_mlir.compile(mlp, x, output_type="linalg-on-tensors")
# print(module)

# Outputs from the model
# logits = mlp(x)

script_model = torch.jit.script(mlp)
# https://pytorch.org/docs/stable/jit.html#interpreting-graphs
graph = script_model.graph
# Print out Torch script graph
# print(graph)

code = script_model.code
# Print out Torch script code
# print(code)

# Print out ONNX graph
# torch.onnx.export(mlp, x, "mlp.onnx", verbose=True, input_names=["actual_input"], output_names=["actual_output"])
