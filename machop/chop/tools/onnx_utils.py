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

ONNX_TO_TORCH_DTYPE = [
    torch.float16,
    torch.float,
    torch.double,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64, 
    torch.uint8,
    torch.float, # uint16
    torch.float, # uint32
    torch.float, # uint64
    torch.bool,
    torch.float, # string
    torch.bfloat16,
    torch.float, # float8e4m3fn
    torch.float, # float8e4m3fnuz
    torch.float, # float8e5m2
    torch.float, # float8e5m2fnuz
    torch.float, # uint4
    torch.float  # int4
]

def onnx_to_torch_dtype(dtype):
    """
        Return Pytorch equivalent of ONNX data type.
        When an equivalent is not available, return float.
    """
    return ONNX_TO_TORCH_DTYPE[dtype]

def to_tensor(input):
    """
        ONNX operators can be infinitely linked but Torch operators cannot
        For example, torch.nn.functional.size() returns a torch.Size instance, which cannot be passed
            to another function that expects a torch.Tensor. So here we convert function outputs to tensors
    """
    if isinstance(input, torch.Tensor):
        return input
    if isinstance(input, torch.Size):
        return torch.Tensor(list(input))


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
    "Erf": {"fx_op": "call_function", "target": torch.special.erf, "input_mapping": ["input"], "input_transform": [None]},
    "Tanh": {"fx_op": "call_function", "target": torch.tanh},
    "LessOrEqual": {"fx_op": "call_function", "target": torch.le},
    "Min": {"fx_op": "call_function", "target": torch.min},
    "Neg": {"fx_op": "call_function", "target": torch.neg},
    "Log": {"fx_op": "call_function", "target": torch.log},
    "Slice": {"fx_op": "call_function", "target": slice},
    "Gemm": {"fx_op": "call_function", "target": gemm},
    "ConstantOfShape": {"fx_op": "call_function", "target": torch.full},
    "ReduceMean": {"fx_op": "call_function", "target": torch.mean, "input_mapping": ["input", "dim"], "attribute_mapping": ["keepdim", ""], "input_transform": [None, None], "attribute_transform" : [None, None]},
    "Expand": {"fx_op": "call_method", "target": "expand"},
    "Shape": {"fx_op": "call_method", "target": "size", "input_mapping": ["self"], "attribute_mapping": [], , "input_transform": [None], "attribute_transform" : []},
    "Cast": {"fx_op": "call_method", "target": "to", "input_mapping": ["self"], "attribute_mapping": ["", "dtype"], , "input_transform": [None], "attribute_transform" : [None, onnx_to_torch_dtype]},
    "Gather": {"fx_op": "call_function", "target": torch.gather, "input_mapping": ["input", "index"], "attribute_mapping": ["dim"], "input_transform": [to_tensor, None], "attribute_transform" : [None]},
    # Unsqueeze ONNX doc says "axes" is an input, but it's actually an attribute
    "Unsqueeze": {"fx_op": "call_function", "target": torch.unsqueeze, "input_mapping": ["input"], "attribute_mapping": ["dim"], , "input_transform": [None], "attribute_transform" : [None]},
    "Concat": {"fx_op": "call_function", "target": torch.cat, "input_mapping": ["tensors"], "attribute_mapping": ["dim"], , "input_transform": [None], "attribute_transform" : [None]},
    "Reshape": {"fx_op": "call_function", "target": torch.reshape, "input_mapping": ["input", "shape"], "attribute_mapping": [""], , "input_transform": [None, None], "attribute_transform" : [None]},
    "Add": {"fx_op": "call_function", "target": torch.add, "input_mapping": ["input", "other"], "input_transform": [None, None],},
    "Sub": {"fx_op": "call_function", "target": torch.sub, "input_mapping": ["input", "other"], "input_transform": [None, None],},
    "Squeeze": {"fx_op": "call_function", "target": torch.squeeze, "input_mapping": ["input", "dim"], "input_transform": [None, None],},
    "Range": {"fx_op": "call_function", "target": torch.arange, "input_mapping": ["start", "end", "step"], "input_transform": [None, None],},
    "Less": {"fx_op": "call_function", "target": torch.lt, "input_mapping": ["input", "other"], "input_transform": [None, None],},
    "Where": {"fx_op": "call_function", "target": torch.where, "input_mapping": ["condition", "input", "other"], "input_transform": [None, None, None],},
    "Mul": {"fx_op": "call_function", "target": torch.mul, "input_mapping": ["input", "other"], "input_transform": [None, None],},
    "Equal": {"fx_op": "call_function", "target": torch.eq, "input_mapping": ["input", "other"], "input_transform": [None, None],},
    "CumSum": {"fx_op": "call_function", "target": torch.cumsum, "input_mapping": ["input", "other"], "input_transform": [None, None],},
    "Pow": {"fx_op": "call_function", "target": torch.pow, "input_mapping": ["input", "exponent"], "input_transform": [None, None],},
    "Sqrt": {"fx_op": "call_function", "target": torch.sqrt, "input_mapping": ["input"], "input_transform": [None],},
    "Div": {"fx_op": "call_function", "target": torch.div, "input_mapping": ["input", "other"], "input_transform": [None, None],},
    "MatMul": {"fx_op": "call_function", "target": torch.matmul, "input_mapping": ["input", "other"], "input_transform": [None, None],},
    "Transpose": {"fx_op": "call_function", "target": torch.transpose},
    "Max": {"fx_op": "call_function", "target": torch.max, "input_mapping": ["input"], "input_transform": [None],},
    "Softmax": {"fx_op": "call_function", "target": torch.nn.functional.softmax, "input_mapping": ["input"], "attribute_mapping": ["dim"], , "input_transform": [None], "attribute_transform" : [None]},
    "Relu": {"fx_op": "call_function", "target": torch.relu, "input_mapping": ["input"], "input_transform": [None],},
}
