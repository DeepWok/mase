import logging
from ....utils import get_mase_op, get_mase_type

logger = logging.getLogger(__name__)


OPERANDS_TO_META_ARG_NAMES = {
    "add": {
        "required": (("data_in", "data_in"), ("data_in_0", "data_in_1")),
        "optional": None,
    },
    "bmm": {
        "required": (("data_in", "weight"), ("data_in_0", "data_in_1")),
        "optional": None,
    },
    "conv1d": {
        "required": (("data_in", "weight"), ("data_in_0", "weight")),
        "optional": (("bias",), ("bias",)),
    },
    "conv2d": {
        "required": (("data_in", "weight"), ("data_in_0", "weight")),
        "optional": (("bias",), ("bias",)),
    },
    "matmul": {
        "required": (("data_in", "weight"), ("data_in_0", "data_in_1")),
        "optional": None,
    },
    "mul": {
        "required": (("data_in", "data_in"), ("data_in_0", "data_in_1")),
        "optional": None,
    },
    "linear": {
        "required": (("data_in", "weight"), ("data_in_0", "weight")),
        "optional": (("bias",), ("bias",)),
    },
    "relu": {
        "required": (("data_in",), ("data_in_0",)),
        "optional": None,
    },
    "sub": {
        "required": (("data_in", "data_in"), ("data_in_0", "data_in_1")),
        "optional": None,
    },
}


def update_node_meta_param_fixed(node, q_config):
    """Add fixed-point precision to node meta for quantization

    Precision format: [width, frac_width]
    """
    mase_op = get_mase_op(node)
    if mase_op not in OPERANDS_TO_META_ARG_NAMES:
        raise ValueError(
            f"Unsupported MASE operation name `{mase_op}` for updating node meta for quantization"
        )

    required_args = OPERANDS_TO_META_ARG_NAMES[mase_op]["required"]
    optional_args = OPERANDS_TO_META_ARG_NAMES[mase_op]["optional"]

    for operand_name, arg_name in zip(*required_args):
        node.meta["mase"].parameters["common"]["args"][arg_name]["type"] = "fixed"
        node.meta["mase"].parameters["common"]["args"][arg_name]["precision"] = [
            q_config[f"{operand_name}_width"],
            q_config[f"{operand_name}_frac_width"],
        ]

    if optional_args is not None:
        for operand_name, arg_name in zip(*optional_args):
            if arg_name in node.meta["mase"].parameters["common"]["args"]:
                if not (
                    f"{operand_name}_width" in q_config
                    and f"{operand_name}_frac_width" in q_config
                ):
                    raise RuntimeError(
                        f"Optional argument {arg_name} found in node meta, but not found in q_config: {q_config}"
                    )
                node.meta["mase"].parameters["common"]["args"][arg_name][
                    "type"
                ] = "fixed"
                node.meta["mase"].parameters["common"]["args"][arg_name][
                    "precision"
                ] = [
                    q_config[f"{operand_name}_width"],
                    q_config[f"{operand_name}_frac_width"],
                ]


def relink_node_meta(node, model):
    node.meta["mase"].node = node
    node.meta["mase"].model = model


def update_q_meta_param(node, config: dict):
    q_arith = config["name"]

    match q_arith:
        case "fixed":
            update_node_meta_param_fixed(node, config)
        case _:
            raise ValueError(f"Unsupported quantization arithmetic name: {q_arith}")


from torch.fx import Node


def find_next_compute_node(node: Node):
    for n in node.users:
        if get_mase_type(n) in ["module_related_func", "builtin_func"]:
            return node, n
    for n in node.users:
        return find_next_compute_node(n)
    return None, None


def find_prev_compute_node(node: Node):
    for n in node.all_input_nodes:
        if get_mase_type(n) in ["module_related_func", "builtin_func"]:
            return node, n
    for n in node.all_input_nodes:
        return find_prev_compute_node(n)
    return None, None


def infer_result_dtype_and_precision(node: Node):
    """
    ```text
      n_1  n_2
        \  /
        node
    ```

    assign node's args precision & dtype to n_1, n_2 results
    """

    if get_mase_type(node) == "placeholder":
        # input node
        input_node, next_node = find_next_compute_node(node)
        if input_node is None:
            logger.warning(
                f"Failed to find next module_related_func node for input node {node.name}. Check if the graph contains module_related_func"
            )
            return
        i = 0
        for n in next_node.all_input_nodes:
            if n is input_node:
                break
            i += 1
        arg_key = list(next_node.meta["mase"].parameters["common"]["args"].keys())[i]
        arg_value = next_node.meta["mase"].parameters["common"]["args"][arg_key]

        for result in node.meta["mase"].parameters["common"]["results"]:
            node.meta["mase"].parameters["common"]["results"][result]["type"] = (
                arg_value["type"]
            )
            node.meta["mase"].parameters["common"]["results"][result]["precision"] = (
                arg_value["precision"]
            )

        for arg in node.meta["mase"].parameters["common"]["args"]:
            node.meta["mase"].parameters["common"]["args"][arg]["type"] = arg_value[
                "type"
            ]
            node.meta["mase"].parameters["common"]["args"][arg]["precision"] = (
                arg_value["precision"]
            )

        logger.debug(
            f"Inferred arg & result dtype and precision for input node `{node.name}` using `{next_node.name}`"
        )

    elif get_mase_type(node) in ["module_related_func", "builtin_func"]:
        input_node, next_node = find_next_compute_node(node)
        if next_node is None:
            # this is the last compute node in the graph, use its args to infer dtype and precision
            max_precision = None
            max_dtype = None
            max_bitwidth = 0
            for arg in node.meta["mase"].parameters["common"]["args"]:
                if not isinstance(
                    node.meta["mase"].parameters["common"]["args"][arg], dict
                ):
                    continue
                if (
                    not "precision"
                    in node.meta["mase"].parameters["common"]["args"][arg]
                ):
                    continue
                cur_width = node.meta["mase"].parameters["common"]["args"][arg][
                    "precision"
                ][0]
                if cur_width > max_bitwidth:
                    max_bitwidth = cur_width
                    max_precision = node.meta["mase"].parameters["common"]["args"][arg][
                        "precision"
                    ]
                    max_dtype = node.meta["mase"].parameters["common"]["args"][arg][
                        "type"
                    ]

            if max_precision is None:
                raise RuntimeError(
                    f"Failed to infer dtype and precision for module_related_func node {node.name}"
                )

            for result in node.meta["mase"].parameters["common"]["results"]:
                node.meta["mase"].parameters["common"]["results"][result][
                    "type"
                ] = max_dtype
                node.meta["mase"].parameters["common"]["results"][result][
                    "precision"
                ] = max_precision
            logger.debug(
                f"Inferred result dtype and precision for module_related_func node `{node.name}` using its args"
            )
        else:
            # use next compute node's args to infer dtype and precision
            i = 0
            for n in next_node.all_input_nodes:
                if n is input_node:
                    break
                i += 1
            arg_key = list(next_node.meta["mase"].parameters["common"]["args"].keys())[
                i
            ]
            arg_value = next_node.meta["mase"].parameters["common"]["args"][arg_key]

            for result in node.meta["mase"].parameters["common"]["results"]:
                node.meta["mase"].parameters["common"]["results"][result]["type"] = (
                    arg_value["type"]
                )
                node.meta["mase"].parameters["common"]["results"][result][
                    "precision"
                ] = arg_value["precision"]
            logger.debug(
                f"Inferred result dtype and precision for module_related_func node `{node.name}` using `{next_node.name}`"
            )

    elif get_mase_type(node) == "implicit_func":
        input_node, next_node = find_next_compute_node(node)
        user_node, prev_node = find_prev_compute_node(node)

        if next_node is not None:
            i = 0
            for n in next_node.all_input_nodes:
                if n is input_node:
                    break
                i += 1
            arg_key = list(next_node.meta["mase"].parameters["common"]["args"].keys())[
                i
            ]
            arg_value = next_node.meta["mase"].parameters["common"]["args"][arg_key]

            for result in node.meta["mase"].parameters["common"]["results"]:
                node.meta["mase"].parameters["common"]["results"][result]["type"] = (
                    arg_value["type"]
                )
                node.meta["mase"].parameters["common"]["results"][result][
                    "precision"
                ] = arg_value["precision"]

            for arg in node.meta["mase"].parameters["common"]["args"]:
                if not isinstance(
                    node.meta["mase"].parameters["common"]["args"][arg], dict
                ):
                    continue
                if (
                    not "precision"
                    in node.meta["mase"].parameters["common"]["args"][arg]
                ):
                    continue
                node.meta["mase"].parameters["common"]["args"][arg]["type"] = arg_value[
                    "type"
                ]
                node.meta["mase"].parameters["common"]["args"][arg]["precision"] = (
                    arg_value["precision"]
                )
            logger.debug(
                f"Inferred arg & result dtype and precision for implicit_func node `{node.name}` using `{next_node.name}`"
            )

        elif prev_node is not None:
            i = 0
            for n in prev_node.users:
                if n is user_node:
                    break
                i += 1
            arg_key = list(prev_node.meta["mase"].parameters["common"]["args"].keys())[
                i
            ]
            arg_value = prev_node.meta["mase"].parameters["common"]["args"][arg_key]

            for result in node.meta["mase"].parameters["common"]["results"]:
                node.meta["mase"].parameters["common"]["results"][result]["type"] = (
                    arg_value["type"]
                )
                node.meta["mase"].parameters["common"]["results"][result][
                    "precision"
                ] = arg_value["precision"]

            for arg in node.meta["mase"].parameters["common"]["args"]:
                node.meta["mase"].parameters["common"]["args"][arg]["type"] = arg_value[
                    "type"
                ]
                node.meta["mase"].parameters["common"]["args"][arg]["precision"] = (
                    arg_value["precision"]
                )
            logger.debug(
                f"Inferred arg & result dtype and precision for implicit_func node `{node.name}` using `{prev_node.name}`"
            )

        else:
            raise RuntimeError(
                f"Failed to infer dtype and precision for implicit_func node {node.name} as it has no input nodes or users of type `module_related_func`"
            )

    elif get_mase_type(node) == "output":
        # output node
        # find the max precision of all input nodes
        user_node, prev_node = find_prev_compute_node(node)

        if prev_node is None:
            raise RuntimeError(
                f"Failed to find prev module_related_func node for output node {node.name}"
            )

        max_precision = None
        max_dtype = None
        max_bitwidth = 0

        i = 0
        for n in prev_node.users:
            if n is user_node:
                break
            i += 1

        arg_key = list(prev_node.meta["mase"].parameters["common"]["args"].keys())[i]
        arg_value = prev_node.meta["mase"].parameters["common"]["args"][arg_key]

        for result in node.meta["mase"].parameters["common"]["results"]:
            node.meta["mase"].parameters["common"]["results"][result]["type"] = (
                arg_value["type"]
            )
            node.meta["mase"].parameters["common"]["results"][result]["precision"] = (
                arg_value["precision"]
            )

        for arg in node.meta["mase"].parameters["common"]["args"]:
            node.meta["mase"].parameters["common"]["args"][arg]["type"] = arg_value[
                "type"
            ]
            node.meta["mase"].parameters["common"]["args"][arg]["precision"] = (
                arg_value["precision"]
            )

        logger.debug(
            f"Inferred dtype and precision for output node `{node.name}` using `{prev_node.name}`"
        )

    else:
        raise RuntimeError(
            f"Unsupported node type {get_mase_type(node)} for inferring dtype and precision"
        )
