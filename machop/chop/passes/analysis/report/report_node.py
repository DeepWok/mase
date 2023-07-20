import logging
from pprint import pformat
from tabulate import tabulate
from pathlib import Path

logger = logging.getLogger(__name__)


def graph_iterator_inspect_node_type(graph):
    headers = ["Node name", "Fx Node op", "Mase type", "Mase op", "Value type"]
    rows = []
    for node in graph.fx_graph.nodes:
        rows.append(
            [
                node.name,
                node.op,
                node.meta["mase"].parameters["common"]["mase_type"],
                node.meta["mase"].parameters["common"]["mase_op"],
                "NA"
                if "data_in_0" not in node.meta["mase"].parameters["common"]["args"]
                else node.meta["mase"].parameters["common"]["args"]["data_in_0"][
                    "type"
                ],
            ]
        )
    logger.info("Inspecting graph [add_common_node_type_analysis_pass]")
    logger.info("\n" + tabulate(rows, headers=headers))
    return graph


def report_node_type_analysis_pass(graph, pass_args=None):
    """
    Inspect mase graph after initialization/loading, including

    - node inspection on types
    """
    graph = graph_iterator_inspect_node_type(graph)
    return graph


def graph_iterator_inspect_node_shape(graph):
    logger.info("Inspecting graph [add_common_node_shape_analysis_pass]")
    buffer = ""
    for node in graph.fx_graph.nodes:
        buffer += f"{node.name}:\nin:\n"
        for key, value in node.meta["mase"].parameters["common"]["args"].items():
            from_name = "none" if value["from"] is None else value["from"].name
            buffer += "{} = {}, from = {}\n".format(key, value["size"], from_name)
        buffer += "out:\n"
        for key, value in node.meta["mase"].parameters["common"]["results"].items():
            buffer += "{} = {}\n".format(key, value["size"])
        buffer += "\n"
    logger.info(buffer)
    return graph


def report_node_shape_analysis_pass(graph, pass_args=None):
    """
    Inspect mase graph after initialization/loading, including

    - node inspection on shapes
    """
    graph = graph_iterator_inspect_node_shape(graph)
    return graph


def graph_iterator_inspect_node_hardware_type(graph):
    headers = ["Node name", "Fx Node op", "Type", "Tool Chain"]
    rows = []
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        rows.append(
            [
                node.name,
                node.op,
                node.meta["mase"].parameters["common"]["results"]["data_out_0"]["type"],
                node.meta["mase"].parameters["hardware"]["toolchain"],
            ]
        )
    logger.info("Inspecting graph [add_common_node_hardware_type_analysis_pass]")
    logger.info("\n" + tabulate(rows, headers=headers))
    return graph


def report_node_hardware_type_analysis_pass(graph, pass_args=None):
    """
    Inspect mase graph after initialization/loading, including

    - node inspection on hardware types
    """
    graph = graph_iterator_inspect_node_hardware_type(graph)
    return graph


def report_node_meta_param_analysis_pass(graph, pass_args: dict = None):
    """
    Inspect mse metadata.parameter

    pass_args: a dict of arguments for this pass, including
    - "which": a list of options in ["all", "common", "hardware", "software"], default ["all"]
    - "save_path": a str of path to save the table, default None
    """
    which_param = pass_args.get("which", ("all",))
    assert isinstance(which_param, (list, tuple))
    for param in which_param:
        assert param in [
            "all",
            "common",
            "hardware",
            "software",
        ], f"Invalid which_param {param}, must be a list of options in ['all', 'common', 'hardware', 'software'], got {param}"
    save_path = pass_args.get("save_path", None)

    headers = [
        "Node name",
        "Fx Node op",
        "Mase type",
        "Mase op",
    ]

    if "common" in which_param or "all" in which_param:
        headers.append("Common Param")
    if "hardware" in which_param or "all" in which_param:
        headers.append("Hardware Param")
    if "software" in which_param or "all" in which_param:
        headers.append("Software Param")

    rows = []
    for node in graph.fx_graph.nodes:
        new_row = [
            node.name,
            node.op,
            node.meta["mase"].parameters["common"]["mase_type"],
            node.meta["mase"].parameters["common"]["mase_op"],
        ]

        if "common" in which_param or "all" in which_param:
            new_row.append(pformat(node.meta["mase"].parameters["common"]))
        if "hardware" in which_param or "all" in which_param:
            new_row.append(pformat(node.meta["mase"].parameters["hardware"]))
        if "software" in which_param or "all" in which_param:
            new_row.append(pformat(node.meta["mase"].parameters["software"]))

        rows.append(new_row)

    table_txt = tabulate(rows, headers=headers, tablefmt="grid")
    logger.info("Inspecting graph [add_common_meta_param_analysis_pass]")
    logger.info("\n" + table_txt)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(Path(save_path), "w") as f:
            f.write(table_txt)
            logger.info(f"Node meta param table is saved to {save_path}")
    return graph
