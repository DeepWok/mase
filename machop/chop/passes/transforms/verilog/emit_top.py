import glob
import logging
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Process, Queue

import torch
import torch.fx

from chop.passes.utils import vf, v2p, init_project

logger = logging.getLogger(__name__)


def _remove_last_comma(string):
    return string[0 : string.rfind(",")]


def get_input_name(from_node, to_node):
    if from_node == to_node:
        return "data_in_0"
    for key, val in to_node.meta["mase"].parameters["common"]["args"].items():
        if val["from"].name == from_node.name:
            return key
    assert False, f"Cannot find edge from {from_node.name} to {to_node.name}"


def _get_cast_parameters(from_node, to_node, is_start=False, is_end=False):
    assert not (
        is_start and is_end
    ), "An edge cannot start and end both through external signals."
    dout = from_node.meta["mase"].parameters["common"]["results"]["data_out_0"]
    from_type = dout["type"]
    from_prec = dout["precision"]
    arg_name = get_input_name(from_node, to_node)
    to_type = to_node.meta["mase"].parameters["common"]["args"][arg_name]["type"]
    to_prec = to_node.meta["mase"].parameters["common"]["args"][arg_name]["precision"]
    from_name = f"{from_node}_data_out_0"  # TODO: This will need to fix in the future. I put _0 here as it seems to that it only considers data_out_0
    to_name = f"{to_node}_{arg_name}"
    from_param = f"{from_node}_OUT_0"  # TODO: This will need to fix in the future. I put _0 here as it seems to that it only considers data_out_0
    to_param = f"{to_node}_{v2p(arg_name)}"

    if is_start:
        to_type = from_type
        to_prec = from_prec
        assert len(to_node.all_input_nodes) == 1
        from_name = "data_in"
        from_param = "IN"
    if is_end:
        from_type = to_type
        from_prec = to_prec
        to_name = "data_out"
        to_param = "OUT"
    return (
        from_name,
        from_type,
        from_prec,
        from_param,
        to_name,
        to_type,
        to_prec,
        to_param,
    )


def _iterator_load_width_parameters_to_map(node_name, val_list, parameter_map):
    for val, param in val_list.items():
        # TODO: Ignore constant for now - To be encoded into parameters or scalar inputs
        val = v2p(val)
        if param["type"] == "float":
            parameter_map[f"{node_name}_{val}_WIDTH"] = param["precision"][0]
        elif param["type"] == "fixed":
            # Unverified...
            parameter_map[f"{node_name}_{val}_WIDTH"] = param["precision"][0]
            parameter_map[f"{node_name}_{val}_FRAC_WIDTH"] = param["precision"][1]
        elif param["type"] == "binary":
            # TODO: Binary quant
            # For binary we will need to contract the precision for weight to 1. e.g "fc1_WEIGHT_WIDTH"
            # For now we assume this would be set by config file
            # Unverified...
            parameter_map[f"{node_name}_{val}_WIDTH"] = param["precision"][0]
            parameter_map[f"{node_name}_{val}_FRAC_WIDTH"] = param["precision"][1]
        else:
            assert False, "Unknown type: {} {}".format(node_name, param["type"])
    return parameter_map


def _load_width_parameters_to_map(graph, parameter_map):
    """
    Add width information to the global parameter map
    """

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        node_name = vf(node.name)

        args = node.meta["mase"].parameters["common"]["args"]
        parameter_map = _iterator_load_width_parameters_to_map(
            node_name, args, parameter_map
        )
        results = node.meta["mase"].parameters["common"]["results"]
        parameter_map = _iterator_load_width_parameters_to_map(
            node_name, results, parameter_map
        )
    return parameter_map


def _load_verilog_parameters_to_map(graph, parameter_map):
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        node_name = vf(node.name)

        for key, value in (
            node.meta["mase"].parameters["hardware"]["verilog_parameters"].items()
        ):
            if not isinstance(value, (int, float, complex, bool)):
                value = '"' + value + '"'
            parameter_map[f"{node_name}_{key}"] = value
    return parameter_map


def _emit_parameters_top(graph):
    """
    Emit parameters at the top-level for the top-level module
    """

    nodes_in = graph.nodes_in
    nodes_out = graph.nodes_out
    node_in_name = vf(nodes_in[0].name)
    node_out_name = vf(nodes_out[0].name)

    parameters = ""
    parameter_map = {}
    parameter_map = _load_width_parameters_to_map(graph, parameter_map)
    parameter_map = _load_verilog_parameters_to_map(graph, parameter_map)

    for key, value in parameter_map.items():
        parameters += f"parameter {key} = {value},\n"

    # Top-level design interface
    parameters += f"""
parameter IN_WIDTH = {node_in_name}_IN_WIDTH,
parameter OUT_WIDTH = {node_out_name}_OUT_WIDTH,
parameter IN_SIZE = {node_in_name}_IN_SIZE,
parameter OUT_SIZE = {node_out_name}_OUT_SIZE,
"""
    return parameters, parameter_map


def _emit_interface_top(graph, parameter_map):
    """
    Emit interface signal declarations for the top-level module
    """

    # Assume the model always has a single input and single output
    interface = """
input  [IN_WIDTH-1:0] data_in [IN_SIZE-1:0],
input  data_in_valid,
output data_in_ready,
output [OUT_WIDTH-1:0] data_out [OUT_SIZE-1:0],
output data_out_valid,
input  data_out_ready,
"""
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        node_name = vf(node.name)
        for key, value in node.meta["mase"].parameters["common"]["results"].items():
            if key == "data_out_0":
                continue
            cap_key = v2p(key)
            print(parameter_map)
            width = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
            size = parameter_map[f"{node_name}_{cap_key}_SIZE"]
            debug_info = f"// [{width}][{size}]"
            interface += f"""{debug_info}
output [{node_name}_{cap_key}_WIDTH-1:0] {node_name}_{key} [{node_name}_{cap_key}_SIZE-1:0],
output {node_name}_{key}_valid,
input  {node_name}_{key}_ready,
"""
    return interface


def _emit_signals_top_internal(node, parameter_map):
    signals = ""
    node_name = vf(node.name)
    # Input signals
    for key, value in node.meta["mase"].parameters["common"]["args"].items():
        # No internal signals if the memory is stored off chip
        if (
            "data_in" not in key
            and node.meta["mase"].parameters["hardware"]["interface_parameters"][key][
                "storage"
            ]
            != "BRAM"
        ):
            continue
        # TODO: Ignore constant arg
        if "value" in value.keys():
            continue
        cap_key = v2p(key)
        width = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
        size = parameter_map[f"{node_name}_{cap_key}_SIZE"]
        debug_info = f"// [{width}][{size}]"
        signals += f"""{debug_info}
logic [{node_name}_{cap_key}_WIDTH-1:0]  {node_name}_{key}        [{node_name}_{cap_key}_SIZE-1:0];
logic                             {node_name}_{key}_valid;
logic                             {node_name}_{key}_ready;
"""

    # Output signals
    for key, value in node.meta["mase"].parameters["common"]["results"].items():
        # No internal signals if the memory is stored off chip
        if (
            key != "data_out_0"
            and node.meta["mase"].parameters["hardware"]["interface_parameters"][key][
                "storage"
            ]
            != "BRAM"
        ):
            continue
        cap_key = v2p(key)
        width = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
        size = parameter_map[f"{node_name}_{cap_key}_SIZE"]
        debug_info = f"// [{width}][{size}]"
        signals += f"""{debug_info}
logic [{node_name}_{cap_key}_WIDTH-1:0]  {node_name}_{key}        [{node_name}_{cap_key}_SIZE-1:0];
logic                             {node_name}_{key}_valid;
logic                             {node_name}_{key}_ready;
"""

    return signals


def _emit_signals_top_hls(node, parameter_map):
    node_name = vf(node.name)
    # Control signals for HLS component
    signals = f"""
logic {node_name}_start;
logic {node_name}_done;
logic {node_name}_idle;
logic {node_name}_ready;
logic {node_name}_ce;
"""

    # Input signals
    for key, value in node.meta["mase"].parameters["common"]["args"].items():
        # No internal signals if the memory is stored off chip
        if (
            "data_in" not in key
            and node.meta["mase"].parameters["hardware"]["interface_parameters"][key][
                "storage"
            ]
            != "BRAM"
        ):
            continue
        cap_key = v2p(key)
        width = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
        size = math.prod(value["size"])
        if key != "data_in":
            debug_info = f"// [{width}][{size}]"
            a_width = math.ceil(math.log2(size))
        else:
            depth = parameter_map[f"{node_name}_{cap_key}_DEPTH"]
            debug_info = f"// [{width}][{depth*size}]"
            a_width = math.ceil(math.log2(depth * size))
        signals += f"""{debug_info}
logic [{node_name}_{cap_key}_WIDTH-1:0]  {node_name}_{key}_q0;
logic [{a_width}-1:0]                    {node_name}_{key}_address0;
logic                                    {node_name}_{key}_ce0;
"""

    # Output signals
    for key, value in node.meta["mase"].parameters["common"]["results"].items():
        # No internal signals if the memory is stored off chip
        if (
            "data_out" not in key
            and node.meta["mase"].parameters["hardware"]["interface_parameters"][key][
                "storage"
            ]
            != "BRAM"
        ):
            continue
        cap_key = v2p(key)
        width = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
        size = math.prod(value["size"])
        debug_info = f"// [{width}][{size}]"
        a_width = math.ceil(math.log2(size))
        signals += f"""{debug_info}
logic [{node_name}_{cap_key}_WIDTH-1:0]  {node_name}_{key}_d0;
logic [{a_width}-1:0]                    {node_name}_{key}_address0;
logic                                    {node_name}_{key}_ce0;
logic                                    {node_name}_{key}_we0;
"""
    return signals


def _emit_signals_top(graph, parameter_map):
    """
    Emit internal signal declarations for the top-level module
    """

    signals = ""
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        node_name = vf(node.name)
        signals += f"""
// --------------------------
//   {node_name} signals
// --------------------------
"""
        if "INTERNAL" in node.meta["mase"].parameters["hardware"]["toolchain"]:
            signals += _emit_signals_top_internal(node, parameter_map)
        elif node.meta["mase"].parameters["hardware"]["toolchain"] == "MLIR_HLS":
            signals += _emit_signals_top_hls(node, parameter_map)
        else:
            assert False, "Unknown node toolchain for signal declarations."

    return signals


def _emit_components_top_internal(node, parameter_map):
    node_name = vf(node.name)

    # Emit kernel instance
    parameters = ""
    for key, value in (
        node.meta["mase"].parameters["hardware"]["verilog_parameters"].items()
    ):
        key_value = parameter_map[f"{node_name}_{key}"]
        debug_info = f"// = {key_value}"
        parameters += f".{key}({node_name}_{key}), {debug_info}\n"
    parameters = _remove_last_comma(parameters)
    component_name = node.meta["mase"].parameters["hardware"]["module"]
    signals = ""
    for key, value in node.meta["mase"].parameters["common"]["args"].items():
        cap_key = v2p(key)
        width = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
        size = (
            parameter_map[f"{node_name}_{cap_key}_SIZE"]
            if "value" not in value.keys()
            else 1
        )
        debug_info = f"// [{width}][{size}]"
        signals += f"""
.{key}({node_name}_{key}), {debug_info}
.{key}_valid({node_name}_{key}_valid),
.{key}_ready({node_name}_{key}_ready),
"""

    for key, value in node.meta["mase"].parameters["common"]["results"].items():
        cap_key = v2p(key)
        width = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
        size = parameter_map[f"{node_name}_{cap_key}_SIZE"]
        debug_info = f"// [{width}][{size}]"
        signals += f"""
.{key}({node_name}_{key}), {debug_info}
.{key}_valid({node_name}_{key}_valid),
.{key}_ready({node_name}_{key}_ready),
"""
    signals = _remove_last_comma(signals)
    components = f"""
// {node_name}
{component_name} #(
{parameters}
) {node_name}_inst (
.clk(clk),
.rst(rst),
{signals}
);
"""

    # Emit parameter instance
    for key, value in node.meta["mase"].parameters["common"]["args"].items():
        # Skip the parameter instance if the memory is stored off chip
        if (
            "data_in" in key
            or node.meta["mase"].parameters["hardware"]["interface_parameters"][key][
                "storage"
            ]
            != "BRAM"
        ):
            continue
        components += _emit_parameters_top_internal(key, value, node, parameter_map)

    for key, value in node.meta["mase"].parameters["common"]["results"].items():
        # Skip the parameter instance if the memory is stored off chip
        if (
            key == "data_out_0"
            or node.meta["mase"].parameters["hardware"]["interface_parameters"][key][
                "storage"
            ]
            != "BRAM"
        ):
            continue
        components += _emit_parameters_top_internal(key, value, node, parameter_map)

    return components


def _emit_components_top_hls(node, parameter_map):
    node_name = vf(node.name)

    # Emit kernel instance
    component_name = node.meta["mase"].parameters["hardware"]["module"]
    signals = ""
    for key, value in node.meta["mase"].parameters["common"]["args"].items():
        cap_key = v2p(key)
        width = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
        size = math.prod(value["size"])
        if "data_in" not in key:
            debug_info = f"// [{width}][{size}]"
        else:
            depth = size
            debug_info = f"// [{width}][{depth*size}]"
        signals += f"""
.{key}_address0({node_name}_{key}_address0), {debug_info}
.{key}_ce0({node_name}_{key}_ce0),
.{key}_q0({node_name}_{key}_q0),
"""

    for key, value in node.meta["mase"].parameters["common"]["results"].items():
        cap_key = v2p(key)
        width = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
        size = math.prod(value["size"])
        debug_info = f"// [{width}][{size}]"
        signals += f"""
.{key}_address0({node_name}_{key}_address0), {debug_info}
.{key}_ce0({node_name}_{key}_ce0),
.{key}_we0({node_name}_{key}_we0),
.{key}_d0({node_name}_{key}_d0),
"""
    signals = _remove_last_comma(signals)
    components = f"""
// {node_name}
{component_name} #(
) {node_name}_inst (
.ap_clk(clk),
.ap_rst(rst),
.ap_start({node_name}_start),
.ap_idle({node_name}_idle),
.ap_ready({node_name}_ready),
.ap_done({node_name}_done),
.ap_ce({node_name}_ce),
{signals}
);
"""

    # Emit parameter instance
    for key, value in node.meta["mase"].parameters["common"]["args"].items():
        # Skip the parameter instance if the memory is stored off chip
        if (
            "data_in" in key
            or node.meta["mase"].parameters["hardware"]["interface_parameters"][key][
                "storage"
            ]
            != "BRAM"
        ):
            continue
        components += _emit_parameters_top_hls(key, value, node, parameter_map)

    for key, value in node.meta["mase"].parameters["common"]["results"].items():
        # Skip the parameter instance if the memory is stored off chip
        if (
            "data_out" in key
            or node.meta["mase"].parameters["hardware"]["interface_parameters"][key][
                "storage"
            ]
            != "BRAM"
        ):
            continue
        components += _emit_parameters_top_hls(key, value, node, parameter_map)

    return components


def _emit_components_top(graph, parameter_map):
    """
    Emit component declarations for the top-level module
    """

    components = """
// --------------------------
//   Kernel instantiation
// --------------------------
"""
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        if "INTERNAL" in node.meta["mase"].parameters["hardware"]["toolchain"]:
            components += _emit_components_top_internal(node, parameter_map)
        elif node.meta["mase"].parameters["hardware"]["toolchain"] == "MLIR_HLS":
            components += _emit_components_top_hls(node, parameter_map)
        else:
            assert False, "Unknown node toolchain for signal declarations."

    return components


def _emit_parameters_top_hls(key, value, node, parameter_map):
    node_name = vf(node.name)
    cap_key = v2p(key)
    component_name = f"{node_name}_{key}_source"
    component_name_inst = f"{node_name}_{key}_0"
    size_debug_info = math.prod(value["size"])
    a_width = math.ceil(math.log2(size_debug_info))
    width_debug_info = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
    return f"""
{component_name} #(
.DATA_WIDTH({node_name}_{cap_key}_WIDTH), // = {width_debug_info}
.ADDR_RANGE({node_name}_{cap_key}_SIZE), // = {size_debug_info}
.ADDR_WIDTH({a_width})
) {component_name_inst} (
.clk(clk),
.reset(rst),
.address0({node_name}_{key}_address0),
.ce0({node_name}_{key}_ce0),
.q0({node_name}_{key}_q0)
);
"""


def _emit_parameters_top_internal(key, value, node, parameter_map):
    node_name = vf(node.name)
    cap_key = v2p(key)
    component_name = f"{node_name}_{key}_source"
    component_name_inst = f"{component_name}_0"
    width_debug_info = parameter_map[f"{node_name}_{cap_key}_WIDTH"]
    size_debug_info = parameter_map[f"{node_name}_{cap_key}_SIZE"]
    key_debug_info = "[{}][{}]".format(
        parameter_map[f"{node_name}_{cap_key}_WIDTH"],
        parameter_map[f"{node_name}_{cap_key}_SIZE"],
    )
    if key == "bias":
        depth = 1
        depth_debug_info = 1
    else:
        depth = f"{node_name}_IN_0_DEPTH"
        depth_debug_info = parameter_map[f"{node_name}_IN_0_DEPTH"]

    return f"""
{component_name} #(
.OUT_DEPTH({depth}), // = {depth_debug_info}
.OUT_WIDTH({node_name}_{cap_key}_WIDTH), // = {width_debug_info}
.OUT_SIZE({node_name}_{cap_key}_SIZE) // = {size_debug_info}
) {component_name_inst} (
.clk(clk),
.rst(rst),
.data_out({node_name}_{key}), // {key_debug_info}
.data_out_ready({node_name}_{key}_ready),
.data_out_valid({node_name}_{key}_valid)
);
"""


def _emit_hs_wires_top(from_node, to_node, parameter_map, is_start=False, is_end=False):
    (
        from_name,
        to_name,
        from_param,
        to_param,
        cast_name,
        data_cast,
    ) = _cast_data(
        "", from_node, to_node, parameter_map, is_start=is_start, is_end=is_end
    )

    return f"""
{data_cast}
assign {from_name}_ready  = {to_name}_ready;
assign {to_name}_valid    = {from_name}_valid;
assign {to_name} = {cast_name};
"""


def _emit_hs2bram_wires_top(
    from_node, to_node, parameter_map, is_start=False, is_end=False
):
    # Add IP files

    (
        from_name,
        to_name,
        from_param,
        to_param,
        cast_name,
        data_cast,
    ) = _cast_data(
        "", from_node, to_node, parameter_map, is_start=is_start, is_end=is_end
    )

    to_node_name = vf(to_node.name)
    from_node_name = vf(from_node.name)
    depth = math.prod(
        to_node.meta["mase"].parameters["common"]["args"]["data_in_0"]["size"]
    )
    size = 1
    width = parameter_map[f"{to_node_name}_IN_0_WIDTH"]
    if is_start:
        in_size = size
    else:
        in_size = parameter_map[f"{from_node_name}_OUT_SIZE"]
    a_width = math.ceil(math.log2(depth * size))
    return f"""
{data_cast}
hs2bram_cast #(
.IN_SIZE({from_param}_SIZE), // = {in_size}
.IN_WIDTH({to_param}_WIDTH), // = {width}
.ADDR_RANGE({to_param}_DEPTH*{to_param}_SIZE), // = {depth*size}
.ADDR_WIDTH({a_width})
) {from_name}_{to_name}_hs2bram_cast (
.data_in_ready({from_name}_ready),
.data_in({cast_name}),
.data_in_valid({from_name}_valid),
.address0({to_name}_address0),
.ce0({to_name}_ce0),
.q0({to_name}_q0),
.out_start({to_node_name}_start),
.out_ready({to_node_name}_ready),
.clk(clk),
.rst(rst)
);
"""


def _emit_bram2hs_wires_top(
    from_node, to_node, parameter_map, is_start=False, is_end=False
):
    (
        from_name,
        to_name,
        from_param,
        to_param,
        cast_name,
        data_cast,
    ) = _cast_data(
        "_d0", from_node, to_node, parameter_map, is_start=is_start, is_end=is_end
    )

    to_node_name = vf(to_node.name)
    from_node_name = vf(from_node.name)
    size = 1
    width = parameter_map[f"{to_node_name}_OUT_0_WIDTH"]
    if is_end:
        out_size = size
    else:
        out_size = parameter_map[f"{to_node_name}_IN_0_SIZE"]
    a_width = math.ceil(math.log2(size))
    return f"""
{data_cast}
bram2hs_cast #(
.OUT_SIZE({from_param}_SIZE), // = {out_size}
.OUT_WIDTH({to_param}_WIDTH), // = {width}
.ADDR_RANGE({to_param}_SIZE), // = {size}
.ADDR_WIDTH({a_width})
) {from_name}_{to_name}_bram2hs_cast (
.address0({from_name}_address0),
.ce0({from_name}_ce0),
.we0({from_name}_we0),
.d0({cast_name}),
.data_out_ready({to_name}_ready),
.data_out({to_name}),
.data_out_valid({to_name}_valid),
.in_done({from_node_name}_done),
.in_ce({from_node_name}_ce),
.clk(clk),
.rst(rst)
);
"""


def _cast_data(
    from_name_tag, from_node, to_node, parameter_map, is_start=False, is_end=False
):
    (
        from_name,
        from_type,
        from_prec,
        from_param,
        to_name,
        to_type,
        to_prec,
        to_param,
    ) = _get_cast_parameters(from_node, to_node, is_start=is_start, is_end=is_end)
    cast_name = f"{from_name}{from_name_tag}"
    data_cast = ""

    if from_type == "fixed" and to_type == "fixed" and from_prec != to_prec:
        in_width = parameter_map[f"{from_param}_WIDTH"]
        in_size = parameter_map[f"{from_param}_SIZE"]
        out_width = parameter_map[f"{to_param}_WIDTH"]
        out_size = parameter_map[f"{to_param}_SIZE"]
        debug_info_in = f"// [{in_width}][{in_size}]"
        debug_info_out = f"// [{out_width}][{out_size}]"
        from_name_cast = cast_name
        cast_name = f"{from_name_cast}_cast"
        data_cast = f"""// assign {cast_name} = {from_name_cast}
logic [{from_param}_WIDTH-1:0] {cast_name} [{from_param}_SIZE-1:0];
fixed_cast #(
    .IN_SIZE(1),
    .IN_WIDTH({from_param}_WIDTH),
    .IN_FRAC_WIDTH({from_param}_FRAC_WIDTH),
    .OUT_WIDTH({to_param}_WIDTH),
    .OUT_FRAC_WIDTH({to_param}_FRAC_WIDTH)
) {from_name}_{to_name}_cast (
    .data_in ({from_name_cast}), {debug_info_out}
    .data_out({cast_name}) {debug_info_in}
);
"""

    elif (from_type == "fixed" or from_type == "binary") and to_type == "float":
        in_width = parameter_map[f"{from_param}_WIDTH"]
        in_size = parameter_map[f"{from_param}_SIZE"]
        out_width = parameter_map[f"{to_param}_WIDTH"]
        out_size = parameter_map[f"{to_param}_SIZE"]
        in_frac_width = parameter_map[f"{from_param}_FRAC_WIDTH"]
        debug_info_in = f"// [{in_width}][{in_size}] frac_width = {in_frac_width}"
        debug_info_out = f"// [{out_width}][{out_size}]"
        from_name_cast = cast_name
        cast_name = f"{from_name_cast}_cast"
        data_cast = f"""// assign {cast_name} = {from_name_cast}
logic [{from_param}_WIDTH-1:0] {cast_name} [{from_param}_SIZE-1:0];
fixed_to_float_cast #(
    .IN_SIZE({from_param}_SIZE),
    .IN_WIDTH({from_param}_WIDTH),
    .IN_FRAC_WIDTH({from_param}_FRAC_WIDTH),
    .OUT_WIDTH({to_param}_WIDTH),
) {from_name}_{to_name}_cast (
    .data_in ({from_name_cast}), {debug_info_out}
    .data_out({cast_name}) {debug_info_in}
);
"""
        # TODO: Added bitcast_op

    elif from_type == "float" and (to_type == "fixed" or to_type == "binary"):
        in_width = parameter_map[f"{from_param}_WIDTH"]
        in_size = parameter_map[f"{from_param}_SIZE"]
        out_width = parameter_map[f"{to_param}_WIDTH"]
        out_size = parameter_map[f"{to_param}_SIZE"]
        out_frac_width = parameter_map[f"{to_param}_FRAC_WIDTH"]
        debug_info_in = f"// [{in_width}][{in_size}]"
        debug_info_out = f"// [{out_width}][{out_size}] frac_width = {out_frac_width} "
        from_name_cast = cast_name
        cast_name = f"{from_name_cast}_cast"
        data_cast = f"""// assign {cast_name} = {from_name_cast}
logic [{from_param}_WIDTH-1:0] {cast_name} [{from_param}_SIZE-1:0];
float_to_fixed_cast #(
    .IN_SIZE({from_param}_SIZE),
    .IN_WIDTH({from_param}_WIDTH),
    .OUT_FRAC_WIDTH({to_param}_FRAC_WIDTH),
    .OUT_WIDTH({to_param}_WIDTH),
) {from_name_cast}_{to_name}_cast (
    .data_in ({from_name_cast}), {debug_info_out}
    .data_out({cast_name}) {debug_info_in}
);
"""
        # TODO: Added bitcast_op

    elif from_type == to_type and from_prec == to_prec:
        pass
    else:
        assert (
            False
        ), f"Unsupported type conversion. Maybe {from_type} != {to_type} or {from_prec} != {to_prec}"

    return from_name, to_name, from_param, to_param, cast_name, data_cast


def _emit_bram_wires_top(
    from_node, to_node, parameter_map, is_start=False, is_end=False
):
    (
        from_name,
        to_name,
        from_param,
        to_param,
        cast_name,
        data_cast,
    ) = _cast_data(
        "_d0", from_node, to_node, parameter_map, is_start=is_start, is_end=is_end
    )

    to_node_name = vf(to_node.name)
    from_node_name = vf(from_node.name)
    depth = math.prod(
        to_node.meta["mase"].parameters["common"]["args"]["data_in_0"]["size"]
    )
    size = 1
    width = parameter_map[f"{to_param}_WIDTH"]
    a_width = math.ceil(math.log2(depth))
    return f"""
{data_cast}
bram_cast #(
.IN_WIDTH({to_param}_WIDTH), // = {width}
.ADDR_RANGE({to_param}_DEPTH*{to_param}_SIZE), // = {depth*size}
.ADDR_WIDTH({a_width})
) {cast_name}_{to_name}_bram_cast (
.address1({from_name}_address0),
.ce1({from_name}_ce0),
.we1({from_name}_we0),
.d1({cast_name}[0]),
.in_done({from_node_name}_done),
.in_ce({from_node_name}_ce),
.address0({to_name}_address0),
.ce0({to_name}_ce0),
.q0({to_name}_q0),
.out_start({to_node_name}_start),
.out_ready({to_node_name}_ready),
.clk(clk),
.rst(rst)
);
"""


def _emit_wires_top(graph, parameter_map):
    """
    Emit internal signal connections for the top-level module
    This includes two interconnection types:
    1. Type casting between inputs and outputs
    2. Interface casting between inputs and outputs
    """

    wires = """
// --------------------------
//   Interconnections
// --------------------------
"""

    nodes_in = graph.nodes_in
    nodes_out = graph.nodes_out
    node_in_name = vf(nodes_in[0].target)
    arg_in = nodes_in[0].meta["mase"].parameters["common"]["args"]["data_in_0"]
    in_type = arg_in["type"]
    in_prec = arg_in["precision"]
    if "INTERNAL" in nodes_in[0].meta["mase"].parameters["hardware"]["toolchain"]:
        wires += _emit_hs_wires_top(
            nodes_in[0], nodes_in[0], parameter_map, is_start=True
        )
    elif nodes_in[0].meta["mase"].parameters["hardware"]["toolchain"] == "MLIR_HLS":
        wires += _emit_hs2bram_wires_top(
            nodes_in[0], nodes_in[0], parameter_map, is_start=True
        )
    else:
        assert False, "Unknown node toolchain for signal declarations."
    node_out_name = vf(nodes_out[0].target)
    out_type = (
        nodes_out[0].meta["mase"].parameters["common"]["results"]["data_out_0"]["type"]
    )
    out_prec = (
        nodes_out[0]
        .meta["mase"]
        .parameters["common"]["results"]["data_out_0"]["precision"]
    )
    if "INTERNAL" in nodes_out[0].meta["mase"].parameters["hardware"]["toolchain"]:
        wires += _emit_hs_wires_top(
            nodes_out[0], nodes_out[0], parameter_map, is_end=True
        )
    elif nodes_out[0].meta["mase"].parameters["hardware"]["toolchain"] == "MLIR_HLS":
        wires += _emit_bram2hs_wires_top(
            nodes_out[0], nodes_out[0], parameter_map, is_end=True
        )
    else:
        assert False, "Unknown node toolchain for signal declarations."

    searched = []
    while nodes_in != nodes_out:
        next_nodes_in = []
        for node in nodes_in:
            node_name = vf(node.name)
            node_tc = node.meta["mase"].parameters["hardware"]["toolchain"]
            if node not in searched:
                searched.append(node)
            for next_node, x in node.users.items():
                if next_node.meta["mase"].parameters["hardware"]["is_implicit"]:
                    if next_node not in next_nodes_in:
                        next_nodes_in.append(node)
                    continue
                else:
                    if next_node not in next_nodes_in and next_node not in searched:
                        next_nodes_in.append(next_node)
                next_node_name = vf(next_node.name)
                next_node_tc = next_node.meta["mase"].parameters["hardware"][
                    "toolchain"
                ]
                if "INTERNAL" in node_tc and "INTERNAL" in next_node_tc:
                    wires += _emit_hs_wires_top(node, next_node, parameter_map)
                elif node_tc == "MLIR_HLS" and "INTERNAL" in next_node_tc:
                    wires += _emit_bram2hs_wires_top(node, next_node, parameter_map)
                elif "INTERNAL" in node_tc and next_node_tc == "MLIR_HLS":
                    wires += _emit_hs2bram_wires_top(node, next_node, parameter_map)
                elif node_tc == "MLIR_HLS" and next_node_tc == "MLIR_HLS":
                    wires += _emit_bram_wires_top(node, next_node, parameter_map)
                else:
                    assert False, "Unknown node toolchain for signal declarations."
        assert (
            nodes_in != next_nodes_in
        ), f"Parsing error: cannot find the next nodes: {nodes_in}."
        nodes_in = next_nodes_in
    return wires


def emit_top(graph, top_name):
    parameters_to_emit, parameter_map = _emit_parameters_top(graph)
    logger.debug(parameter_map)
    parameters_to_emit = _remove_last_comma(parameters_to_emit)
    interface_to_emit = _emit_interface_top(graph, parameter_map)
    interface_to_emit = _remove_last_comma(interface_to_emit)
    signals_to_emit = _emit_signals_top(graph, parameter_map)
    components_to_emit = _emit_components_top(graph, parameter_map)
    wires_to_emit = _emit_wires_top(graph, parameter_map)
    time_to_emit = time.strftime("%d/%m/%Y %H:%M:%S")
    module_inst = """
// =====================================
//     Mase Hardware
//     Model: {}
//     {}
// =====================================
`timescale 1ns/1ps
module {} #({}
) (
input clk,
input rst,
{}
);
{}
{}
{}
endmodule
""".format(
        top_name,
        time_to_emit,
        top_name,
        parameters_to_emit,
        interface_to_emit,
        signals_to_emit,
        components_to_emit,
        wires_to_emit,
    )
    return module_inst


def emit_verilog_top_transform_pass(graph, pass_args={}):
    """
    Emit the top-level model deisgn in Verilog
    """

    logger.info("Emitting Verilog...")
    project_dir = (
        pass_args["project_dir"] if "project_dir" in pass_args.keys() else "top"
    )
    top_name = pass_args["top_name"] if "top_name" in pass_args.keys() else "top"

    init_project(project_dir)
    rtl_dir = os.path.join(project_dir, "hardware", "rtl")

    top = emit_top(graph, top_name)

    top_file = os.path.join(rtl_dir, f"{top_name}.sv")
    top_design = open(top_file, "w")
    top_design.write(top)
    top_design.close()
    return graph
