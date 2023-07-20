import torch
from ...utils import get_mase_op

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------


def _set_arg_param(mase_meta, arg_name, key, value):
    if "software" not in mase_meta.parameters:
        mase_meta.parameters["software"] = {}
    if "args" not in mase_meta.parameters["software"]:
        mase_meta.parameters["software"]["args"] = {}
    if arg_name not in mase_meta.parameters["software"]["args"]:
        mase_meta.parameters["software"]["args"][arg_name] = {}

    mase_meta.parameters["software"]["args"][arg_name][key] = value
    return mase_meta


def _set_result_param(mase_meta, result_name, key, value):
    if "software" not in mase_meta.parameters:
        mase_meta.parameters["software"] = {}
    if "results" not in mase_meta.parameters["software"]:
        mase_meta.parameters["software"]["results"] = {}
    if result_name not in mase_meta.parameters["software"]["results"]:
        mase_meta.parameters["software"]["results"][result_name] = {}

    mase_meta.parameters["software"]["results"][result_name][key] = value
    return mase_meta


# ------------------------------------------------------------------------------
# Initialization of mase metadata software parameters
# ------------------------------------------------------------------------------

PASS_DEFAULTS = {
    "stat": {},  # analysis/statistical_profiler
}


def analyze_software_meta_param_nn_module_default(meta):
    """
    *: module
    Initialize the default software meta parameters for a nn.Module
    """
    assert isinstance(
        meta.module, torch.nn.Module
    ), f"meta.module must be a nn.Module, got {type(meta.module)}"

    for pass_name, default_value in PASS_DEFAULTS.items():
        _set_arg_param(meta, "data_in_0", pass_name, default_value)

        for name, _ in meta.module.named_parameters():
            _set_arg_param(meta, name, pass_name, default_value)

        for name, _ in meta.module.named_buffers():
            _set_arg_param(meta, name, pass_name, default_value)

        _set_result_param(meta, "data_out_0", pass_name, default_value)


FUNCTIONAL_ENTRY_MAP = {
    "adaptive_avg_pool1d": None,
    "adaptive_avg_pool2d": None,
    "adaptive_max_pool1d": None,
    "adaptive_max_pool2d": None,
    "avg_pool1d": None,
    "avg_pool2d": None,
    "batch_norm": {
        "data_in_1": "weight",
        "data_in_2": "bias",
        "data_in_3": "running_mean",
        "data_in_4": "running_var",
    },
    "conv1d": {"data_in_1": "weight", "data_in_2": "bias"},
    "conv2d": {"data_in_1": "weight", "data_in_2": "bias"},
    "layer_norm": {"data_in_1": "weight", "data_in_2": "bias"},
    "linear": {"data_in_1": "weight", "data_in_2": "bias"},
    "max_pool1d": None,
    "max_pool2d": None,
    "relu": None,
}


def analyze_software_meta_param_nn_functional_default(meta):
    """
    Initialize the default software meta parameters for a nn.functional
    """
    node = meta.node
    op = get_mase_op(node)

    for pass_name, default_value in PASS_DEFAULTS.items():
        for i in range(len(node.all_input_nodes)):
            arg_entry = f"data_in_{i}"
            if (
                FUNCTIONAL_ENTRY_MAP.get(op) is not None
                and arg_entry in FUNCTIONAL_ENTRY_MAP[op]
            ):
                arg_entry = FUNCTIONAL_ENTRY_MAP[op][arg_entry]
            _set_arg_param(meta, arg_entry, pass_name, default_value)

        _set_result_param(meta, "data_out_0", pass_name, default_value)


def analyze_software_meta_param_module_related_func_default(meta):
    """
    *: module related func
    Initialize the default software meta parameters for a module related function
    """
    if meta.module is not None:
        analyze_software_meta_param_nn_module_default(meta)
    else:
        analyze_software_meta_param_nn_functional_default(meta)


def analyze_software_meta_param_nn_module_batch_norm(meta):
    analyze_software_meta_param_nn_module_default(meta)
    meta.parameters["software"]["args"].pop("num_batches_tracked")


def analyze_software_meta_param_builtin_func_default(meta):
    """
    *: builtin func
    Initialize the default software meta parameters for a builtin function
    """
    node = meta.node

    for pass_name, default_value in PASS_DEFAULTS.items():
        for i in range(len(node.all_input_nodes)):
            _set_arg_param(meta, f"data_in_{i}", pass_name, default_value)

        _set_result_param(meta, "data_out_0", pass_name, default_value)


def analyze_software_meta_param_implicit_func_default(meta):
    """
    *: implicit func
    Initialize the default software meta parameters for a implicit function
    """
    node = meta.node

    for pass_name, default_value in PASS_DEFAULTS.items():
        for i in range(len(node.all_input_nodes)):
            _set_arg_param(meta, f"data_in_{i}", pass_name, default_value)

        if get_mase_op(node) not in ["size"]:
            _set_result_param(meta, "data_out_0", pass_name, default_value)


def analyze_software_meta_param_placeholder(meta):
    """
    *: placeholder
    """
    for pass_name, default_value in PASS_DEFAULTS.items():
        _set_result_param(meta, "data_out_0", pass_name, default_value)


def analyze_software_meta_param_get_attr(meta):
    """
    *: get_attr
    """
    for pass_name, default_value in PASS_DEFAULTS.items():
        _set_result_param(meta, "data_out_0", pass_name, default_value)


def analyze_software_meta_param_output(meta):
    """
    *: output
    """
    for pass_name, default_value in PASS_DEFAULTS.items():
        _set_arg_param(meta, "data_in_0", pass_name, default_value)


# ------------------------------------------------------------------------------

SOFTWARE_PARAM_ANALYSIS_LAYERS = {
    "module": {
        "batch_norm1d": analyze_software_meta_param_nn_module_batch_norm,
        "batch_norm2d": analyze_software_meta_param_nn_module_batch_norm,
        # default:
        "default": analyze_software_meta_param_nn_module_default,
    },
    "module_related_func": {
        "adaptive_avg_pool1d": analyze_software_meta_param_module_related_func_default,
        "adaptive_avg_pool2d": analyze_software_meta_param_module_related_func_default,
        "adaptive_max_pool1d": analyze_software_meta_param_module_related_func_default,
        "adaptive_max_pool2d": analyze_software_meta_param_module_related_func_default,
        "avg_pool1d": analyze_software_meta_param_module_related_func_default,
        "avg_pool2d": analyze_software_meta_param_module_related_func_default,
        "batch_norm": analyze_software_meta_param_module_related_func_default,
        "conv1d": analyze_software_meta_param_module_related_func_default,
        "conv2d": analyze_software_meta_param_module_related_func_default,
        "layer_norm": analyze_software_meta_param_module_related_func_default,
        "linear": analyze_software_meta_param_module_related_func_default,
        "max_pool1d": analyze_software_meta_param_module_related_func_default,
        "max_pool2d": analyze_software_meta_param_module_related_func_default,
        "relu": analyze_software_meta_param_module_related_func_default,
        # default:
        "default": analyze_software_meta_param_module_related_func_default,
    },
    # builtin func
    "builtin_func": {
        "mul": analyze_software_meta_param_builtin_func_default,
        "sub": analyze_software_meta_param_builtin_func_default,
        "add": analyze_software_meta_param_builtin_func_default,
        "matmul": analyze_software_meta_param_builtin_func_default,
        "bmm": analyze_software_meta_param_builtin_func_default,
        # default:
        "default": analyze_software_meta_param_builtin_func_default,
    },
    "implicit_func": {
        "size": analyze_software_meta_param_implicit_func_default,
        "view": analyze_software_meta_param_implicit_func_default,
        "flatten": analyze_software_meta_param_implicit_func_default,
        "t": analyze_software_meta_param_implicit_func_default,
        "constant": analyze_software_meta_param_implicit_func_default,
    },
    "placeholder": {
        "placeholder": analyze_software_meta_param_placeholder,
    },
    "get_attr": {
        "get_attr": analyze_software_meta_param_get_attr,
    },
    "output": {
        "output": analyze_software_meta_param_output,
    },
}
