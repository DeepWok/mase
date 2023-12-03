import torch


def gemm(A, B, C=None, alpha=1.0, beta=1.0, transA=False, transB=False):
    # Transpose matrices A and B if needed
    A = A.transpose() if transA else A
    B = B.transpose() if transB else B

    # Perform matrix multiplication
    result = alpha * torch.matmul(A, B)

    # Add optional matrix C
    if C is not None:
        result += beta * C

    return result


def slice(data, starts, ends, axes=None, steps=None):
    # Convert data to a PyTorch tensor if it's not already
    data = torch.tensor(data)

    # Get the rank (number of dimensions) of the input tensor
    r = len(data.shape)

    # Handle default values for axes and steps
    if axes is None:
        axes = list(range(r))
    if steps is None:
        steps = [1] * len(starts)

    # Convert negative axes to non-negative values
    axes = [a + r if a < 0 else a for a in axes]

    # Initialize effective values
    start = [0] * r
    end = list(data.shape)
    step = [1] * r

    # Adjust values based on starts, ends, axes, and steps
    for i in range(len(starts)):
        start[axes[i]] = starts[i]
        end[axes[i]] = ends[i]
        step[axes[i]] = steps[i]

    # Clamp values for negative stepping
    for i in range(len(starts)):
        if step[i] < 0:
            start[i] = max(0, min(data.shape[i] - 1, start[i]))
            end[i] = max(-1, min(data.shape[i], end[i]))
        else:
            start[i] = max(0, min(data.shape[i], start[i]))
            end[i] = max(0, min(data.shape[i], end[i]))

    # Generate slices based on adjusted values
    slices = [slice(start[i], end[i], step[i]) for i in range(r)]

    # Perform the slicing operation
    result = data[slices]

    return result


def identity(data):
    id = torch.nn.Identity()
    return id(data)


def conv(x, w, b, auto_pad, dilations, group, kernel_shape, pads, strides):
    # TO DO
    conv = torch.nn.Conv2d()
    return conv(x)


def instance_norm_2d(input, scale, b, epsilon):
    # TO DO
    norm = torch.nn.InstanceNorm2d(1)
    return norm(input)


ONNX_OP_MAPPING = {
    "Greater": {"fx_op": "call_function", "target": torch.gt},
    "Abs": {"fx_op": "call_function", "target": torch.abs},
    "Sin": {"fx_op": "call_function", "target": torch.sin},
    "Cos": {"fx_op": "call_function", "target": torch.cos},
    "Sigmoid": {"fx_op": "call_function", "target": torch.sigmoid},
    "ArgMax": {"fx_op": "call_function", "target": torch.argmax},
    "Flatten": {"fx_op": "call_function", "target": torch.flatten},
    "Split": {"fx_op": "call_function", "target": torch.split},
    "Not": {"fx_op": "call_function", "target": torch.logical_not},
    "Identity": {"fx_op": "call_function", "target": identity},
    "Tile": {"fx_op": "call_function", "target": torch.tile},
    "Erf": {"fx_op": "call_function", "target": torch.special.erf},
    "Tanh": {"fx_op": "call_function", "target": torch.tanh},
    "LessOrEqual": {"fx_op": "call_function", "target": torch.le},
    "Min": {"fx_op": "call_function", "target": torch.min},
    "Neg": {"fx_op": "call_function", "target": torch.neg},
    "Log": {"fx_op": "call_function", "target": torch.log},
    "Slice": {"fx_op": "call_function", "target": slice},
    "Gemm": {"fx_op": "call_function", "target": gemm},
    "ConstantOfShape": {"fx_op": "call_function", "target": torch.full},
    "ReduceMean": {"fx_op": "call_function", "target": torch.mean},
    "Expand": {"fx_op": "call_method", "target": "expand"},
    "Shape": {"fx_op": "call_method", "target": "size"},
    "Cast": {"fx_op": "call_method", "target": "to"},
    "Gather": {"fx_op": "call_function", "target": torch.gather},
    "Unsqueeze": {"fx_op": "call_function", "target": torch.unsqueeze},
    "Concat": {"fx_op": "call_function", "target": torch.cat},
    "Reshape": {"fx_op": "call_function", "target": torch.reshape},
    "Add": {"fx_op": "call_function", "target": torch.add},
    "Sub": {"fx_op": "call_function", "target": torch.sub},
    "Squeeze": {"fx_op": "call_function", "target": torch.squeeze},
    "Range": {"fx_op": "call_function", "target": torch.arange},
    "Less": {"fx_op": "call_function", "target": torch.lt},
    "Where": {"fx_op": "call_function", "target": torch.where},
    "Mul": {"fx_op": "call_function", "target": torch.mul},
    "Equal": {"fx_op": "call_function", "target": torch.eq},
    "CumSum": {"fx_op": "call_function", "target": torch.cumsum},
    "Pow": {"fx_op": "call_function", "target": torch.pow},
    "Sqrt": {"fx_op": "call_function", "target": torch.sqrt},
    "Div": {"fx_op": "call_function", "target": torch.div},
    "MatMul": {"fx_op": "call_function", "target": torch.matmul},
    "Transpose": {"fx_op": "call_function", "target": torch.transpose},
    "Max": {"fx_op": "call_function", "target": torch.max},
    "Softmax": {"fx_op": "call_function", "target": torch.softmax},
    "Relu": {"fx_op": "call_function", "target": torch.relu},
}
