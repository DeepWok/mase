from ._set_common_dtype import (
    _set_dtype_before_call_function,
    _set_dtype_before_call_method,
    _set_dtype_before_call_module,
)
from ._set_common_empty import (
    _set_empty_metadata_before_call_function,
    _set_empty_metadata_before_call_method,
    _set_empty_metadata_before_call_module,
)
from ._set_common_size import (
    _set_arg_size_before_call_function,
    _set_arg_size_before_call_method,
    _set_arg_size_before_call_module,
    _set_result_size_after_call_function,
    _set_result_size_after_call_method,
    _set_result_size_after_call_module,
)


def set_metadata_common_before_call_function(node, function, args, kwargs):
    _set_empty_metadata_before_call_function(node, function, args, kwargs)
    _set_arg_size_before_call_function(node, function, args, kwargs)
    _set_dtype_before_call_function(node, function, args, kwargs)


def set_metadata_common_after_call_function(node, function, output):
    _set_result_size_after_call_function(node, function, output)


def set_metadata_common_before_call_module(node, module, args, kwargs):
    _set_empty_metadata_before_call_module(node, module, args, kwargs)
    _set_arg_size_before_call_module(node, module, args, kwargs)
    _set_dtype_before_call_module(node, module, args, kwargs)


def set_metadata_common_after_call_module(node, module, output):
    _set_result_size_after_call_module(node, module, output)


def set_metadata_common_before_call_method(node, method_name, args, kwargs):
    _set_empty_metadata_before_call_method(node, method_name, args, kwargs)
    _set_arg_size_before_call_method(node, method_name, args, kwargs)
    _set_dtype_before_call_method(node, method_name, args, kwargs)


def set_metadata_common_after_call_method(node, method_name, output):
    _set_result_size_after_call_method(node, method_name, output)
