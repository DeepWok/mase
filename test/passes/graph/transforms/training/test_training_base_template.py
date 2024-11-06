from functools import partial
from chop.nn.quantizers.integer import IntegerQuantize
from torch import Tensor, nn, optim, manual_seed
from torch.nn import functional as F
from torch.autograd import gradcheck
import torch

manual_seed(0)


# Functional Equivalent of LinearInteger
def linearInteger(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
    out_config: dict = None,
):
    w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
    x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
    b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]

    x = IntegerQuantize.apply(x, x_width, x_frac_width)
    weight = IntegerQuantize.apply(weight, w_width, w_frac_width)
    if bias is not None:
        bias = IntegerQuantize.apply(bias, b_width, b_frac_width)
    out = F.linear(x, weight, bias)

    if out_config is not None:
        out_width, out_frac_width = (
            out_config["data_out_width"],
            out_config["data_out_frac_width"],
        )
        out = IntegerQuantize.apply(out, out_width, out_frac_width)

    return out


# Linearly Quantized Gradient
def linearGradInteger(ctx, grad_output, config: dict = None):
    inputs, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None

    grad_output = IntegerQuantize.apply(
        grad_output, config["output_grad_width"], config["output_grad_frac_width"]
    )
    inputs = IntegerQuantize.apply(
        inputs, config["data_in_width"], config["data_in_frac_width"]
    )
    weight = IntegerQuantize.apply(
        weight, config["weight_width"], config["weight_frac_width"]
    )
    if bias is not None:
        bias = IntegerQuantize.apply(
            bias, config["bias_width"], config["bias_frac_width"]
        )

    if ctx.needs_input_grad[0]:
        grad_input = grad_output.mm(weight)
    if ctx.needs_input_grad[1]:
        grad_weight = grad_output.t().mm(inputs)
    if bias is not None and ctx.needs_input_grad[2]:
        grad_bias = grad_output.sum(0)

    return grad_input, grad_weight, grad_bias


class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(x: Tensor, weight: Tensor, bias: Tensor = None):
        return F.linear(x, weight, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inputs, weight, bias = inputs
        ctx.save_for_backward(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(inputs)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class QLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(QLinear, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.linear_autograd_fn = clone_autograd_fn(CustomLinearFunction)

    def forward(self, x: Tensor):
        return self.linear_autograd_fn.apply(x, self.weight, self.bias)


def clone_autograd_fn(autograd_fn: torch.autograd.Function):
    class ClonedAutogradFn(autograd_fn):
        pass

    return ClonedAutogradFn


def attach_quantized_forward(q_fn: torch.autograd.Function, q_fn_cfg: dict):
    """
    Attach a quantized forward function to a quantized forward function
    """
    name = q_fn_cfg["name"]
    # Redefine from Quantizer level
    if name == "integer":
        q_forward = partial(linearInteger, config=q_fn_cfg)
    else:
        raise ValueError(f"Quantization function {name} not recognized")

    q_fn.forward = q_forward


def attach_quantized_backward(q_fn: torch.autograd.Function, q_fn_cfg: dict):
    """
    Attach a quantized backward function to a quantized forward function
    """
    name = q_fn_cfg["name"]
    if name == "integer":
        q_backward = partial(linearGradInteger, config=q_fn_cfg)
        # q_backward = partial(linearGradInteger, config=q_fn_cfg)
    else:
        raise ValueError(f"Quantization function {name} not recognized")
    q_fn.backward = q_backward


q_cfg = {
    "forward": {
        "pass": "quantization",
        "name": "integer",
        "weight_width": 10,
        "weight_frac_width": 5,
        "data_in_width": 10,
        "data_in_frac_width": 5,
        "bias_width": 10,
        "bias_frac_width": 5,
        "data_out_width": 10,
        "data_out_frac_width": 5,
    },
    "backward": {
        "pass": "quantization",
        "name": "integer",
        "output_grad_width": 10,
        "output_grad_frac_width": 5,
        "data_in_width": 10,
        "data_in_frac_width": 5,
        "weight_width": 10,
        "weight_frac_width": 5,
        "bias_width": 10,
        "bias_frac_width": 5,
    },
}


# ------------------------------------------------
# Validate the un-quanitzed forward function
# ------------------------------------------------
def linear(x: Tensor, weight: Tensor, bias: Tensor = None):
    return CustomLinearFunction.apply(x, weight, bias)


input = (
    torch.randn(20, 10, dtype=torch.double, requires_grad=True),
    torch.randn(5, 10, dtype=torch.double, requires_grad=True),
    torch.randn(5, dtype=torch.double, requires_grad=True),
)
test = gradcheck(linear, input, eps=1e-6, atol=1e-4)


# ------------------------------------------------
# Model Setup for training
# ------------------------------------------------
fc1 = QLinear(5, 10)
fc1.weight.data = torch.randn(5, 10, dtype=torch.double, requires_grad=True)
fc1.bias.data = torch.randn(5, dtype=torch.double, requires_grad=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(fc1.parameters(), lr=0.01, momentum=0.9)
# some random input and lables
inputs = torch.randn(4, 10, dtype=torch.double, requires_grad=True)
labels = torch.rand(4) * 10 // 5
inputs, labels = inputs.to(device), labels.to(device)
fc1.to(device)


# ------------------------------------------------
# Test quantized forward function
# ------------------------------------------------
outputs = fc1(inputs)
attach_quantized_forward(fc1.linear_autograd_fn, q_cfg["forward"])
q_outputs = fc1(inputs)
print(outputs, q_outputs)

# ------------------------------------------------
# Test quantized backward function
# ------------------------------------------------
optimizer.zero_grad()
outputs = fc1(inputs)
loss = criterion(outputs, labels.long())
loss.backward()
print(fc1.weight.grad)
print(fc1.bias.grad)

attach_quantized_backward(fc1.linear_autograd_fn, q_cfg["backward"])

optimizer.zero_grad()
outputs = fc1(inputs)
loss = criterion(outputs, labels.long())
loss.backward()
print(fc1.weight.grad)
print(fc1.bias.grad)
