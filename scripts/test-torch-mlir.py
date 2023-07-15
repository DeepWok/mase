# --------------------------------------------------------------------
#   This example converts a simple MLP model to MLIR to test torch-mlir
# --------------------------------------------------------------------

import os

import torch
import torch.nn as nn
import torch_mlir


# Model specifications
class MLP(torch.nn.Module):
    # Toy quantized FC model for digit recognition on MNIST

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 28 * 28)
        self.fc2 = nn.Linear(28 * 28, 28 * 28 * 4)
        self.fc3 = nn.Linear(28 * 28 * 4, 10)

    def forward(self, x):
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
# For output_type= argument, expected one of: TORCH, LINALG_ON_TENSORS, TOSA, STABLEHLO, RAW
# module = torch_mlir.compile(mlp, x, output_type="stablehlo")
module = torch_mlir.compile(mlp, x, output_type="linalg-on-tensors")
mlir_path = "toy.linalg.mlir"
with open(mlir_path, "w", encoding="utf-8") as outf:
    outf.write(str(module))
print("MLIR of module toy successfully written into ./toy.linalg.mlir")
assert os.path.isfile(mlir_path), "Linalg MLIR generation failed."
