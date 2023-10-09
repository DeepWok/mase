import inspect
import math

import torch
from chop.passes.utils import vf, get_node_by_name

# ----------------------------------------------------------
# Placeholder
# ----------------------------------------------------------


def analyse_common_parameters_placeholder(meta, dummy_in):
    """
    The placeholder itself does not contain any information, but can be provided from users.
    """
    var_name = meta.node.target
    if var_name not in dummy_in.keys():
        raise ValueError(f"Cannot find model input {var_name} in the given module")
    arg = dummy_in[var_name]
    # deal with model specific inputs, normally these are not numerical values/tensors
    if var_name in meta.model.additional_inputs:
        meta.parameters["common"]["args"] = {}
        meta.parameters["common"]["results"] = {}
        meta.parameters["common"]["results"]["data_out_0"] = {
            "type": "model_specific_input",
            "value": meta.model.additional_inputs[var_name],
            "size": [1],
            "value": arg,
        }
        return meta

    meta.parameters["common"]["args"] = {}
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"]["data_out_0"] = {
        "type": "float",
        "precision": [32],
        "size": list(arg.size()),
    }

    return meta


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------


def analyse_common_parameters_output(meta):
    """
    Skip as output has no user
    """
    meta.parameters["common"]["results"] = {}
    assert len(meta.node.users.keys()) == 0
    return meta


# ----------------------------------------------------------
# Linear
# ----------------------------------------------------------


def analyse_common_parameters_linear(meta):
    if meta.module is not None:
        for name, parameter in meta.module.named_parameters():
            meta.parameters["common"]["args"][name] = {
                "type": "float",
                "precision": [32],
                "size": list(parameter.shape),
                "from": None,
            }
    else:
        meta.parameters["common"]["args"]["weight"] = meta.parameters["common"][
            "args"
        ].pop("data_in_1")
        meta.parameters["common"]["args"]["bias"] = meta.parameters["common"][
            "args"
        ].pop("data_in_2")

    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": [
                meta.parameters["common"]["args"]["data_in_0"]["size"][0],
                meta.parameters["common"]["args"]["weight"]["size"][0],
            ],
        }
    }
    return meta


# ----------------------------------------------------------
# ReLU, Binary Op
# ----------------------------------------------------------


def analyse_common_parameters_pass(meta):
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": meta.parameters["common"]["args"]["data_in_0"]["size"],
        }
    }
    return meta


# ----------------------------------------------------------
# Attr
# ----------------------------------------------------------


def _fetch_attr(target: str, meta):
    """
    Get attr return tensor
    """

    target_atoms = target.split(".")
    attr_itr = meta.model
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def analyse_common_parameters_attr(meta):
    """
    A constant is an op that provides constant.
    """

    result = _fetch_attr(meta.node.target, meta)

    meta.parameters["common"]["args"] = {}
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"]["data_out_0"] = {
        "type": "float",
        "precision": [32],
        "size": list(result.size()),
    }
    return meta


# ----------------------------------------------------------
# Method
# ----------------------------------------------------------


def _load_arg(meta):
    return torch.fx.graph.map_arg(
        meta.node.args, lambda n: get_node_by_name(n.meta["mase"].graph, n.name)
    )


def _load_kwarg(meta):
    return torch.fx.graph.map_arg(
        meta.node.kwargs, lambda n: get_node_by_name(n.meta["mase"].graph, n.name)
    )


def _arg_shape_to_val(args):
    args_val = []
    for a in args:
        if isinstance(a, torch.fx.Node):
            if "value" in a.meta["mase"].parameters["common"]["results"]["data_out_0"]:
                val = a.meta["mase"].parameters["common"]["results"]["data_out_0"][
                    "value"
                ]
            else:
                val = torch.full(
                    a.meta["mase"].parameters["common"]["results"]["data_out_0"][
                        "size"
                    ],
                    1.0,
                )
        elif isinstance(a, dict):
            val = {}
            for _k, _a in a.items():
                if isinstance(_a, torch.fx.Node):
                    if (
                        "value"
                        in _a.meta["mase"].parameters["common"]["results"]["data_out_0"]
                    ):
                        val[_k] = _a.meta["mase"].parameters["common"]["results"][
                            "data_out_0"
                        ]["value"]
                    else:
                        val[_k] = torch.full(
                            _a.meta["mase"].parameters["common"]["results"][
                                "data_out_0"
                            ]["size"],
                            1.0,
                        )
                else:
                    val[_k] = _a
        else:
            val = a
        args_val.append(val)
    return args_val


def _kwarg_shape_to_val(kwargs):
    # Some values must be int types but the typing system in MASE is not ready
    # Here we record the variable names that must be int types
    KNOWN_INT_KEYS = ["labels"]

    kwargs_val = {}
    for key, val in kwargs.items():
        if isinstance(val, torch.fx.Node):
            if key in KNOWN_INT_KEYS:
                kwargs_val[key] = torch.full(
                    val.meta["mase"].parameters["common"]["results"]["data_out_0"][
                        "size"
                    ],
                    1,
                )
            else:
                kwargs_val[key] = torch.full(
                    val.meta["mase"].parameters["common"]["results"]["data_out_0"][
                        "size"
                    ],
                    1.0,
                )
        else:
            kwargs_val[key] = val
    return kwargs_val


def analyse_common_parameters_method(meta):
    """
    Memory transformation.
    The current approach is just to run inference and directly fetch the result size.
    TODO: This needs to be replaced with direct shape inference
    """

    self_obj, *args = _load_arg(meta)
    kwargs = _load_kwarg(meta)

    dummy_data = torch.full(
        self_obj.meta["mase"].parameters["common"]["results"]["data_out_0"]["size"], 1
    )
    args_val = _arg_shape_to_val(args)
    kwargs_val = _kwarg_shape_to_val(kwargs)

    result = getattr(dummy_data, meta.node.target)(*args_val, **kwargs_val)

    if isinstance(result, int):
        meta.parameters["common"]["results"] = {}
        arg = meta.parameters["common"]["args"]["data_in_0"]
        meta.parameters["common"]["results"]["data_out_0"] = {
            "type": arg["type"],
            "precision": arg["precision"],
            "size": [1],
            "value": result,
        }
    else:
        size = list(result.size())

        meta.parameters["common"]["results"] = {}
        arg = meta.parameters["common"]["args"]["data_in_0"]
        meta.parameters["common"]["results"]["data_out_0"] = {
            "type": arg["type"],
            "precision": arg["precision"],
            "size": size,
        }

    return meta


# ----------------------------------------------------------
# General Functions
# ----------------------------------------------------------
def _type_check(self_obj, meta, args_val, kwargs_val):
    """
    Obtain the result of the output by executing the function
    """

    is_int = True
    is_float = True
    is_bool = True

    list_depth = lambda L: isinstance(L, list) and max(map(list_depth, L)) + 1
    # Handle case when self_obj is a list of node. E.g. next_node torch.stack
    dummy_data_list = []
    dummy_data_size_list = []

    # This is an annoying bit - Python does not do type casting automatically
    # Here we try both float and int, and pick whatever works
    # TODO: A proper way is to get dummy inputs with the correct type and propagate all along with this pass
    if isinstance(self_obj, list):
        for node in self_obj:
            if (
                "value"
                in node.meta["mase"]
                .parameters["common"]["results"]["data_out_0"]
                .keys()
            ):
                dummy_data = node.meta["mase"].parameters["common"]["results"][
                    "data_out_0"
                ]["value"]
                dummy_data_list.append(dummy_data)
            else:
                dummy_data_size = node.meta["mase"].parameters["common"]["results"][
                    "data_out_0"
                ]["size"]
                dummy_data_size_list.append(dummy_data_size)
        assert not (
            not dummy_data_list and not dummy_data_size_list
        ), "Both dummy_data_size_list and dummy_data_list contains data. This is not supported"

    # handle special case where self_obj is a constant ( e.g. 0 + x ) in this case 0 is the self_obj
    if isinstance(self_obj, int):
        dummy_data_list = self_obj

    if self_obj is None:
        result = meta.node.target(*args_val, **kwargs_val)
    else:
        try:
            if dummy_data_list != []:
                dummy_data = dummy_data_list
            elif dummy_data_size_list:
                dummy_data = []
                for size in dummy_data_size_list:
                    if list_depth(size) > 1:
                        k = size
                        for i in range(0, list_depth(size) - 1):
                            assert len(k) == 1
                            k = k[0]
                        dummy_data.append(
                            torch.full(
                                tuple(k),
                                0,
                            ),
                        )
                    else:
                        dummy_data.append(
                            torch.full(
                                tuple(size),
                                0,
                            )
                        )
            elif (
                "value"
                in self_obj.meta["mase"]
                .parameters["common"]["results"]["data_out_0"]
                .keys()
            ):
                dummy_data = self_obj.meta["mase"].parameters["common"]["results"][
                    "data_out_0"
                ]["value"]
            else:
                size = self_obj.meta["mase"].parameters["common"]["results"][
                    "data_out_0"
                ]["size"]
                # Special tuple input - check relavant comments for single-element tuple result
                if list_depth(size) > 1:
                    k = size
                    for i in range(0, list_depth(size) - 1):
                        assert len(k) == 1
                        k = k[0]
                    dummy_data = (
                        torch.full(
                            tuple(k),
                            0,
                        ),
                    )
                else:
                    dummy_data = torch.full(
                        tuple(size),
                        0,
                    )
            result = meta.node.target(dummy_data, *args_val, **kwargs_val)
        # except RuntimeError:
        except:
            is_int = False

    if not is_int:
        try:
            if dummy_data_list != []:
                dummy_data = dummy_data_list
            elif dummy_data_size_list:
                dummy_data = [
                    (
                        (torch.full(size, 1.0)),
                    )  # Special tuple input - check relavant comments for single-element tuple result
                    if list_depth(size) == 2 and len(size) == 1
                    else torch.full(size, 1.0)
                    for size in dummy_data_size_list
                ]
            elif (
                "value"
                in self_obj.meta["mase"]
                .parameters["common"]["results"]["data_out_0"]
                .keys()
            ):
                dummy_data = self_obj.meta["mase"].parameters["common"]["results"][
                    "data_out_0"
                ]["value"]
            else:
                size = self_obj.meta["mase"].parameters["common"]["results"][
                    "data_out_0"
                ]["size"]
                dummy_data = (
                    (
                        (torch.full(size, 1.0)),
                    )  # Special tuple input - check relavant comments for single-element tuple result
                    if list_depth(size) == 2 and len(size) == 1
                    else torch.full(size, 1.0)
                )
            result = meta.node.target(dummy_data, *args_val, **kwargs_val)
        except:
            is_float = False
    # special handle for torch.where (accept a list of boolean)
    if not is_float:
        try:
            if dummy_data_list != []:
                dummy_data = dummy_data_list
            elif dummy_data_size_list:
                dummy_data = [
                    (
                        torch.full(size, True, dtype=torch.bool),
                    )  # Special tuple input - check relavant comments for single-element tuple result
                    if list_depth(size) == 2 and len(size) == 1
                    else torch.full(size, True, dtype=torch.bool)
                    for size in dummy_data_size_list
                ]
            elif (
                "value"
                in self_obj.meta["mase"]
                .parameters["common"]["results"]["data_out_0"]
                .keys()
            ):
                dummy_data = self_obj.meta["mase"].parameters["common"]["results"][
                    "data_out_0"
                ]["value"]
            else:
                size = self_obj.meta["mase"].parameters["common"]["results"][
                    "data_out_0"
                ]["size"]
                # Special tuple input - check relavant comments for single-element tuple result
                dummy_data = (
                    (
                        torch.full(size, True, dtype=torch.bool),
                    )  # Special tuple input - check relavant comments for single-element tuple result
                    if list_depth(size) == 2 and len(size) == 1
                    else torch.full(size, True, dtype=torch.bool)
                )
            result = meta.node.target(dummy_data, *args_val, **kwargs_val)
        except:
            is_bool = False

    assert (
        is_int or is_float or is_bool
    ), f"Both float and int and bool are not correct for module {meta.node}"

    return result


def _get_result_by_function_simulation(meta):
    """
    Obtain the result of the output by executing the function
    """
    if len(meta.node.args) == 0:
        self_obj = None
        args = []
    else:
        self_obj, *args = _load_arg(meta)

    kwargs = _load_kwarg(meta)
    if isinstance(self_obj, list) or isinstance(self_obj, int):
        pass
    elif (
        self_obj is not None
        and self_obj.meta["mase"].parameters["common"]["results"]["data_out_0"]["size"]
        is None
    ):
        return None

    args_val = _arg_shape_to_val(args)
    kwargs_val = _kwarg_shape_to_val(kwargs)

    result = _type_check(
        self_obj=self_obj, meta=meta, args_val=args_val, kwargs_val=kwargs_val
    )
    return result


def _get_size_by_function_simulation(meta):
    """
    Obtain the size of the output by executing the function
    """
    result = _get_result_by_function_simulation(meta)

    # special handle for function that returns a tensor with size [], 0 or 1 dimension.
    # They should not be considered as a constant.
    if isinstance(result, torch.Tensor) and (
        list(result.size()) == [] or list(result.size()) == [1]
    ):
        return "tensor_with_size_1"

    # special handle for .size()
    if isinstance(result, torch.Size):
        return result
        # return list([i for i in result])
    if isinstance(result, bool) or isinstance(result, int) or isinstance(result, float):
        return [1]

    size = list(result.size())
    return size


# ----------------------------------------------------------
# Size
# ----------------------------------------------------------


def add_meta_for_size(meta, size):
    """
    Size is an attribute which provides a constrant value in a static graph.
    """
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"]["data_out_0"] = {
        "type": "float",
        "precision": [32],
        "size": [len([i for i in size])],
        "value": list([i for i in size]),
    }
    return meta


def _analyse_common_parameters_function_const(meta):
    op = meta.parameters["common"]["mase_op"]
    if op == "x":
        pass
    else:
        assert False, f"Unknown function that returns constant: {op}"

    return meta


def analyse_common_parameters_function(meta):
    if len(meta.node.users) == 0:
        return meta

    size = _get_size_by_function_simulation(meta)

    # special handle for size
    if isinstance(size, torch.Size):
        meta = add_meta_for_size(meta, size)
        return meta
    # special handle for constant
    if size == [1]:
        meta.parameters["common"]["results"] = {
            "data_out_0": {
                "type": "float",
                "precision": [32],
                "size": [1],
                "value": _get_result_by_function_simulation(meta),
            }
        }
        return meta

    # Some functions returns constant values,
    # so special procedures are required here
    # FUNC_RETURN_CONST = ["eq", "getattr", "getitem"]
    # if meta.parameters["common"]["mase_op"] in FUNC_RETURN_CONST:
    #     print(size)
    #     return _analyse_common_parameters_function_const(meta)
    # else:
    #     print(meta.node.target)
    #     print(meta.parameters["common"]["mase_op"])

    # special handle for function that returns a tensor with size [1].
    if size == "tensor_with_size_1":
        size = [1]

    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": size,
        }
    }

    mase_type = meta.parameters["common"]["mase_type"]
    if mase_type != "module_related_func":
        return meta

    # Correct the arg names to match module ops
    mase_op = meta.parameters["common"]["mase_op"]
    if mase_op == "linear":
        meta.parameters["common"]["args"]["weight"] = meta.parameters["common"][
            "args"
        ].pop("data_in_1")
        meta.parameters["common"]["args"]["bias"] = meta.parameters["common"][
            "args"
        ].pop("data_in_2")
    elif mase_op == "relu":
        # data_in_1 is added wrongly because of the inplace functionality of relu {'inplace': False} which counts as an extra Kwargs
        meta.parameters["common"]["args"].pop("data_in_1")
        pass
    # NOTE: We may possibly need to add more cases here...
    elif mase_op == "adaptive_avg_pool2d":
        # Don't do anything as we have no args apart from the actual input
        # See line 218 of chop/passes/analysis/add_metadata/add_common_metadata.py
        # for more information.
        pass
    else:
        assert False, "Unknown module related function - arg names not corrected."

    return meta


# ----------------------------------------------------------
# General Modules
# ----------------------------------------------------------


def _get_size_by_module_simulation(meta):
    """
    Otain the size of the output by executing the module
    """
    self_obj, *args = _load_arg(meta)
    kwargs = _load_kwarg(meta)
    args_val = _arg_shape_to_val(args)
    kwargs_val = _kwarg_shape_to_val(kwargs)

    is_int = True
    is_float = True

    # This is an annoying bit - Python does not do type casting automatically
    # Here we try both float and int, and pick whatever works
    # TODO: A proper way is to get dummy inputs with the correct type and propagate all along with this pass
    try:
        if (
            "value"
            in self_obj.meta["mase"]
            .parameters["common"]["results"]["data_out_0"]
            .keys()
        ):
            dummy_data = self_obj.meta["mase"].parameters["common"]["results"][
                "data_out_0"
            ]["value"]
        else:
            dummy_data = torch.full(
                self_obj.meta["mase"].parameters["common"]["results"]["data_out_0"][
                    "size"
                ],
                0,
            )
        previous_state = meta.module.training
        meta.module.train(False)

        with torch.no_grad():
            result = meta.module(dummy_data, *args_val, **kwargs_val)
    except:
        is_int = False

    if not is_int:
        try:
            if (
                "value"
                in self_obj.meta["mase"]
                .parameters["common"]["results"]["data_out_0"]
                .keys()
            ):
                dummy_data = self_obj.meta["mase"].parameters["common"]["results"][
                    "data_out_0"
                ]["value"]
            else:
                dummy_data = torch.full(
                    self_obj.meta["mase"].parameters["common"]["results"]["data_out_0"][
                        "size"
                    ],
                    1.0,
                )
            previous_state = meta.module.training
            meta.module.train(False)

            with torch.no_grad():
                result = meta.module(dummy_data, *args_val, **kwargs_val)
        # except RuntimeError:
        except:
            is_float = False

    assert (
        is_int or is_float
    ), f"Both float and int are not correct for module {meta.node}"

    meta.module.train(previous_state)

    if isinstance(result, tuple):
        if len(result) != 1:
            assert (
                False
            ), "We have a real tuple output... need to discuss how to deal with it"
        return [[list(result[0].size())]]

    size = self_obj.meta["mase"].parameters["common"]["results"]["data_out_0"]["size"]
    dummy_data = torch.full(size, 1.0)
    size_prods = size[0]
    for i in range(len(size) - 2):
        size_prods *= size[i + 2]
    if "batch_norm" in meta.parameters["common"]["mase_op"] and size_prods == 1:
        return size
        # There is a bug in mase, where batch norms transforms fail because of the fact that
        # size is determined by using a dummy input. When batch size is effectively 1 such as
        # in the case of a dummy input, and the size of the feature layer is 1x1 then BatchNorm fails.
        # This hack bypasses this because batchnorm does not change the shape of its input.
    result = meta.module(dummy_data, *args, **kwargs)
    size = list(result.size())
    return size


def analyse_common_parameters_module(meta):
    for name, parameter in meta.module.named_parameters():
        meta.parameters["common"]["args"][name] = {
            "type": "float",
            "precision": [32],
            "size": list(parameter.shape),
            "from": None,
        }
    size = _get_size_by_module_simulation(meta)

    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": size,
        }
    }
    return meta


# ----------------------------------------------------------
# Conv1d
# ----------------------------------------------------------


def analyse_common_parameters_conv1d(meta):
    if meta.module is not None:
        return analyse_common_parameters_module(meta)
    else:
        assert (
            False
        ), "torch.functional.conv1d not supported. Please use nn.Module.Conv1d for now"
    return meta


# ----------------------------------------------------------
# Conv2d
# ----------------------------------------------------------


def analyse_common_parameters_conv2d(meta):
    if meta.module is not None:
        return analyse_common_parameters_module(meta)
    else:
        assert (
            False
        ), "torch.functional.conv2d not supported. Please use nn.Module.Conv2d for now"

    return meta
