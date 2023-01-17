import torch
from model import MLP

mlp = MLP()

batch_size = 2
x = torch.randn((batch_size, 28, 28))

logits = mlp(x)

script_model = torch.jit.script(mlp)

# https://pytorch.org/docs/stable/jit.html#interpreting-graphs
graph = script_model.graph
# Print out Torch script graph
print(graph)

code = script_model.code
# Print out Torch script code
print(code)

# Print out ONNX graph
torch.onnx.export(mlp, x, "mlp.onnx", verbose=True, input_names=["actual_input"], output_names=["actual_output"])
