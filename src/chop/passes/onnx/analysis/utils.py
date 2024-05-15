import torch
import torch.nn as nn
import torch.fx as fx
import onnx


def clean_name(name):
    node_name = name[1:] if (name[0] == "/") else name
    node_name = node_name.replace("/", "_").replace(".", "_").lower()
    return node_name


def init_submodule(gm, target):
    try:
        # If submodule already exists, do not overwrite
        _ = gm.get_submodule(target)
    except:
        # If it doesn't, initialize submodule
        gm.add_submodule(target, nn.Module())


def deserialize_constant(constant_node):
    assert len(constant_node.attribute) == 1
    tensor_proto = onnx.helper.get_attribute_value(constant_node.attribute[0])
    return torch.from_numpy(onnx.numpy_helper.to_array(tensor_proto))


flatten_list = lambda lst: [
    item
    for sublist in lst
    for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])
]


def add_kwarg(node, kwarg_name, kwarg_value):
    if kwarg_name in node.kwargs:
        # Turn into a list and append
        # node.kwargs[kwarg_name] = [node.kwargs[kwarg_name], kwarg_value]
        new_list = flatten_list([node.kwargs[kwarg_name], kwarg_value])
        node.kwargs = {
            **{
                node.kwargs[key]: value
                for key, value in node.kwargs.items()
                if key != kwarg_name
            },
            kwarg_name: new_list,
        }
    else:
        node.kwargs = {**node.kwargs, kwarg_name: kwarg_value}
