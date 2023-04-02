import glob
import math
import logging
import multiprocessing
import os
import sys
import shutil
import subprocess
import time
from multiprocessing import Process, Queue

import toml
import torch
import torch.fx

from ..graph.mase_graph import MaseGraph
from ..graph.mase_metadata import MaseMetadata
from ..graph.utils import get_module_by_name, vf, v2p
from .mase_mem_emitter import emit_parameters_in_rom_internal
from .mase_mem_emitter import emit_parameters_in_rom_hls

logger = logging.getLogger(__name__)


def _remove_last_comma(string):
    return string[0 : string.rfind(",")]


def _add_dependence_files(files, file_list):
    file_list.extend(f for f in files if f not in file_list)
    return file_list


def _execute(cmd, log_output: bool = True, log_file=None, cwd="."):
    if log_output:
        logger.debug("{} (cwd = {})".format(subprocess.list2cmdline(cmd), cwd))
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, cwd=cwd
        ) as result:
            if log_file:
                f = open(log_file, "w")
            if result.stdout or result.stderr:
                logger.info("")
            if result.stdout:
                for line in result.stdout:
                    if log_file:
                        f.write(line)
                    line = line.rstrip("\n")
                    logging.trace(line)
            if result.stderr:
                for line in result.stderr:
                    if log_file:
                        f.write(line)
                    line = line.rstrip("\n")
                    logging.trace(line)
            if log_file:
                f.close()
    else:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, cwd=cwd)
    return result.returncode


def _get_hls_parameters(node):
    args_param = node.meta.parameters["common"]["args"]
    results_param = node.meta.parameters["common"]["results"]
    hls_param = ""
    for arg, param in args_param.items():
        precision = args_param[arg]["precision"]
        size = str(tuple(args_param[arg]["size"])).replace(",)", ")")
        ty = args_param[arg]["type"]
        hls_param += f"{arg},,in,,{ty},,{size},,{precision};"
    for result, param in results_param.items():
        precision = results_param[result]["precision"]
        size = str(tuple(results_param[result]["size"])).replace(",)", ")")
        ty = results_param[result]["type"]
        hls_param += f"{result},,out,,{ty},,{size},,{precision};"
    return hls_param.replace(" ", "")


def _get_cast_parameters(from_node, to_node, is_start=False, is_end=False):
    assert not (
        is_start and is_end
    ), "An edge cannot start and end both through external siganls."
    from_type = from_node.meta.parameters["common"]["results"]["data_out"]["type"]
    from_prec = from_node.meta.parameters["common"]["results"]["data_out"]["precision"]
    to_type = to_node.meta.parameters["common"]["args"]["data_in"]["type"]
    to_prec = to_node.meta.parameters["common"]["args"]["data_in"]["precision"]
    from_name = f"{from_node}_data_out"
    to_name = f"{to_node}_data_in"
    from_param = f"{from_node}_OUT"
    to_param = f"{to_node}_IN"

    if is_start:
        to_type = from_type
        to_prec = from_prec
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


class MaseVerilogEmitter(MaseGraph):
    def __init__(
        self,
        model=None,
        mode="auto",
        project_dir=None,
        to_debug=False,
        target="xc7z020clg484-1",
        num_targets=1,
        common_param=None,
    ):
        """
        MaseVerilogEmitter loads MaseGraph and emit the corresponding
        hardware design in Verilog.
        model = input model
        mode = synthesizing approach : hls or auto
        common_param = external common parameters from toml (for quantization info)
        project_dir = path of the top-level project
        project = name of the top-level project
        to_debug = whether emit debugging info
        target = the FPGA target for hardware generation
        num_tagrets = number of the FPGA targets
        """

        super().__init__(model=model, common_param=common_param, synth_mode=mode)
        self.root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
        self.project_dir = project_dir
        assert self.project_dir is not None, "Cannot find the project directory"
        if not os.path.isabs(self.project_dir):
            self.project_dir = os.path.join(os.getcwd(), self.project_dir)
        self.project = os.path.basename(self.project_dir)
        self.target = target
        self.num_targets = num_targets
        self.to_debug = to_debug
        self._init_project()
        self.dependence_files = []
        self.verilog_parameters = {}
        if to_debug:
            print(self.fx_graph)

    def _init_project(self):
        project_dir = self.project_dir
        if not os.path.exists(project_dir):
            os.mkdir(project_dir)
        hardware_dir = os.path.join(self.project_dir, "hardware")
        if not os.path.exists(hardware_dir):
            os.mkdir(hardware_dir)
        rtl_dir = os.path.join(hardware_dir, "rtl")
        if not os.path.exists(rtl_dir):
            os.mkdir(rtl_dir)
        else:
            for p in glob.glob(os.path.join(rtl_dir, "*")):
                os.remove(p)

    # ----------------------------------------------------------
    # Emit hardware code
    # ----------------------------------------------------------
    def emit_verilog(self):
        """
        Emit Verilog for unpartitioned model
        """
        self.verify()
        project_dir = self.project_dir
        rtl_dir = os.path.join(project_dir, "hardware", "rtl")
        if not os.path.exists(rtl_dir):
            os.mkdir(rtl_dir)
        # Emit each components in the form of layers
        self.emit_components()
        # Emit the top-level module
        self.emit_top()

        files = ""
        for file in glob.glob(os.path.join(rtl_dir, "*.sv")):
            files += " " + file
        # os.system(f"verilator --lint-only {files}")

        logger.info(f"The hardware design has been successfully generated.")

    def _emit_parameters_top(self):
        """
        Emit parameters at the top-level for the top-level module
        """

        nodes_in = self.nodes_in
        nodes_out = self.nodes_out
        node_in_name = vf(nodes_in[0].name)
        node_out_name = vf(nodes_out[0].name)
        parameters = ""
        for node in self.fx_graph.nodes:
            if node.op != "call_module" and node.op != "call_function":
                continue
            node_name = vf(node.name)
            if (
                node.meta.parameters["hardware"]["toolchain"] == "INTERNAL"
                or node.meta.parameters["hardware"]["toolchain"] == "HLS"
            ):
                for key, value in node.meta.parameters["hardware"][
                    "verilog_parameters"
                ].items():
                    if not isinstance(value, (int, float, complex, bool)):
                        value = '"' + value + '"'
                    parameters += f"parameter {node_name}_{key} = {value},\n"
                    assert f"{node_name}_{key}" not in self.verilog_parameters.keys()
                    self.verilog_parameters[f"{node_name}_{key}"] = value
            elif node.meta.parameters["hardware"]["toolchain"] == "EXTERNAL":
                raise NotImplementedError(f"EXTERNAL not supported yet.")
        parameters += f"""
parameter IN_WIDTH = {node_in_name}_IN_WIDTH,
parameter OUT_WIDTH = {node_out_name}_OUT_WIDTH,
parameter IN_SIZE = {node_in_name}_IN_SIZE,
parameter OUT_SIZE = {node_out_name}_OUT_SIZE,
"""
        return parameters

    def _emit_interface_top(self):
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
        for node in self.fx_graph.nodes:
            if node.op != "call_module" and node.op != "call_function":
                continue
            node_name = vf(node.name)
            for key, value in node.meta.parameters["common"]["args"].items():
                if key == "data_in":
                    continue
                # No top-level interface if the memory is stored on chip
                if (
                    node.meta.parameters["hardware"]["interface_parameters"][key][
                        "storage"
                    ]
                    == "BRAM"
                ):
                    continue
                cap_key = v2p(key)
                width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
                size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
                debug_info = f"// [{width}][{size}]"
                interface += f"""{debug_info}
input  [{node_name}_{cap_key}_WIDTH-1:0] {node_name}_{key} [{node_name}_{cap_key}_SIZE-1:0], 
input  {node_name}_{key}_valid,
output {node_name}_{key}_ready,
"""
            for key, value in node.meta.parameters["common"]["results"].items():
                if key == "data_out":
                    continue
                cap_key = v2p(key)
                width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
                size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
                debug_info = f"// [{width}][{size}]"
                interface += f"""{debug_info}
output [{node_name}_{cap_key}_WIDTH-1:0] {node_name}_{key} [{node_name}_{cap_key}_SIZE-1:0],
output {node_name}_{key}_valid,
input  {node_name}_{key}_ready,
"""
        return interface

    def _emit_siganls_top_internal(self, node):
        signals = ""
        node_name = vf(node.name)
        # Input signals
        for key, value in node.meta.parameters["common"]["args"].items():
            # No internal siganls if the memory is stored off chip
            if (
                key != "data_in"
                and node.meta.parameters["hardware"]["interface_parameters"][key][
                    "storage"
                ]
                != "BRAM"
            ):
                continue
            cap_key = v2p(key)
            width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
            size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
            debug_info = f"// [{width}][{size}]"
            signals += f"""{debug_info}
logic [{node_name}_{cap_key}_WIDTH-1:0]  {node_name}_{key}        [{node_name}_{cap_key}_SIZE-1:0];
logic                             {node_name}_{key}_valid;
logic                             {node_name}_{key}_ready;
"""

        # Output signals
        for key, value in node.meta.parameters["common"]["results"].items():
            # No internal siganls if the memory is stored off chip
            if (
                key != "data_out"
                and node.meta.parameters["hardware"]["interface_parameters"][key][
                    "storage"
                ]
                != "BRAM"
            ):
                continue
            cap_key = v2p(key)
            width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
            size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
            debug_info = f"// [{width}][{size}]"
            signals += f"""{debug_info}
logic [{node_name}_{cap_key}_WIDTH-1:0]  {node_name}_{key}        [{node_name}_{cap_key}_SIZE-1:0];
logic                             {node_name}_{key}_valid;
logic                             {node_name}_{key}_ready;
"""

        return signals

    def _emit_siganls_top_hls(self, node):
        node_name = vf(node.name)
        # Control signals for HLS component
        signals = f"""
logic {node_name}_start;
logic {node_name}_done;
logic {node_name}_idle;
logic {node_name}_ready;
"""

        # Input signals
        for key, value in node.meta.parameters["common"]["args"].items():
            # No internal siganls if the memory is stored off chip
            if (
                key != "data_in"
                and node.meta.parameters["hardware"]["interface_parameters"][key][
                    "storage"
                ]
                != "BRAM"
            ):
                continue
            cap_key = v2p(key)
            width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
            size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
            debug_info = f"// [{width}][{size}]"
            a_width = math.ceil(math.log2(size))
            signals += f"""{debug_info}
logic [{node_name}_{cap_key}_WIDTH-1:0]  {node_name}_{key}_q0;
logic [{a_width}-1:0]                    {node_name}_{key}_address0;
logic                                    {node_name}_{key}_ce0;
"""

        # Output signals
        for key, value in node.meta.parameters["common"]["results"].items():
            # No internal siganls if the memory is stored off chip
            if (
                key != "data_out"
                and node.meta.parameters["hardware"]["interface_parameters"][key][
                    "storage"
                ]
                != "BRAM"
            ):
                continue
            cap_key = v2p(key)
            width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
            size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
            debug_info = f"// [{width}][{size}]"
            a_width = math.ceil(math.log2(size))
            signals += f"""{debug_info}
logic [{node_name}_{cap_key}_WIDTH-1:0]  {node_name}_{key}_d0;
logic [{a_width}-1:0]                    {node_name}_{key}_address0;
logic                                    {node_name}_{key}_ce0;
logic                                    {node_name}_{key}_we0;
"""
        return signals

    def _emit_siganls_top(self):
        """
        Emit internal signal declarations for the top-level module
        """

        signals = ""
        for node in self.fx_graph.nodes:
            if node.op != "call_module" and node.op != "call_function":
                continue
            node_name = vf(node.name)
            signals += f"""
// --------------------------
//   {node_name} signals
// --------------------------
"""
            if node.meta.parameters["hardware"]["toolchain"] == "INTERNAL":
                signals += self._emit_siganls_top_internal(node)
            elif node.meta.parameters["hardware"]["toolchain"] == "HLS":
                signals += self._emit_siganls_top_hls(node)
            else:
                assert False, "Unknown node toolchain for signal declarations."

        return signals

    def _emit_components_top_internal(self, node):
        node_name = vf(node.name)

        # Emit kernel instance
        parameters = ""
        for key, value in node.meta.parameters["hardware"][
            "verilog_parameters"
        ].items():
            key_value = self.verilog_parameters[f"{node_name}_{key}"]
            debug_info = f"// = {key_value}"
            parameters += f".{key}({node_name}_{key}), {debug_info}\n"
        parameters = _remove_last_comma(parameters)
        component_name = node.meta.parameters["hardware"]["module"]
        signals = ""
        for key, value in node.meta.parameters["common"]["args"].items():
            cap_key = v2p(key)
            width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
            size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
            debug_info = f"// [{width}][{size}]"
            signals += f"""
.{key}({node_name}_{key}), {debug_info}
.{key}_valid({node_name}_{key}_valid),
.{key}_ready({node_name}_{key}_ready),
"""

        for key, value in node.meta.parameters["common"]["results"].items():
            cap_key = v2p(key)
            width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
            size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
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
        for key, value in node.meta.parameters["common"]["args"].items():
            # Skip the parameter instance if the memory is stored off chip
            if (
                key == "data_in"
                or node.meta.parameters["hardware"]["interface_parameters"][key][
                    "storage"
                ]
                != "BRAM"
            ):
                continue
            components += self._emit_parameters_top_internal(key, value, node)

        for key, value in node.meta.parameters["common"]["results"].items():
            # Skip the parameter instance if the memory is stored off chip
            if (
                key == "data_out"
                or node.meta.parameters["hardware"]["interface_parameters"][key][
                    "storage"
                ]
                != "BRAM"
            ):
                continue
            components += self._emit_parameters_top_internal(key, value, node)

        return components

    def _emit_components_top_hls(self, node):
        node_name = vf(node.name)

        # Emit kernel instance
        component_name = node.meta.parameters["hardware"]["module"]
        signals = ""
        for key, value in node.meta.parameters["common"]["args"].items():
            cap_key = v2p(key)
            width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
            size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
            debug_info = f"// [{width}][{size}]"
            signals += f"""
.{key}_address0({node_name}_{key}_address0), {debug_info}
.{key}_ce0({node_name}_{key}_ce0),
.{key}_q0({node_name}_{key}_q0),
"""

        for key, value in node.meta.parameters["common"]["results"].items():
            cap_key = v2p(key)
            width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
            size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
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
{signals}
);
"""

        # Emit parameter instance
        for key, value in node.meta.parameters["common"]["args"].items():
            # Skip the parameter instance if the memory is stored off chip
            if (
                key == "data_in"
                or node.meta.parameters["hardware"]["interface_parameters"][key][
                    "storage"
                ]
                != "BRAM"
            ):
                continue
            components += self._emit_parameters_top_hls(key, value, node)

        for key, value in node.meta.parameters["common"]["results"].items():
            # Skip the parameter instance if the memory is stored off chip
            if (
                key == "data_out"
                or node.meta.parameters["hardware"]["interface_parameters"][key][
                    "storage"
                ]
                != "BRAM"
            ):
                continue
            components += self._emit_parameters_top_hls(key, value, node)

        return components

    def _emit_components_top(self):
        """
        Emit component declarations for the top-level module
        """

        components = """
// --------------------------
//   Kernel instantiation 
// --------------------------
"""
        for node in self.fx_graph.nodes:
            if node.op != "call_module" and node.op != "call_function":
                continue
            if node.meta.parameters["hardware"]["toolchain"] == "INTERNAL":
                components += self._emit_components_top_internal(node)
            elif node.meta.parameters["hardware"]["toolchain"] == "HLS":
                components += self._emit_components_top_hls(node)
            else:
                assert False, "Unknown node toolchain for signal declarations."

        return components

    def _emit_parameters_top_hls(self, key, value, node):
        node_name = vf(node.name)
        cap_key = v2p(key)
        component_name = f"{node_name}_{key}_source"
        component_name_inst = f"{node_name}_{key}_0"
        size_debug_info = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
        a_width = math.ceil(math.log2(size_debug_info))
        width_debug_info = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
        return f"""
{component_name} #(
.DATA_WIDTH({node_name}_{cap_key}_WIDTH), // = {width_debug_info}
.ADDR_RANGE({node_name}_IN_DEPTH), // = {size_debug_info}
.ADDR_WIDTH({a_width})
) {component_name_inst} (
.clk(clk),
.reset(rst),
.address0({node_name}_{key}_address0), 
.ce0({node_name}_{key}_ce0),
.q0({node_name}_{key}_q0) 
);
"""

    def _emit_parameters_top_internal(self, key, value, node):
        node_name = vf(node.name)
        cap_key = v2p(key)
        component_name = f"{node_name}_{key}_source"
        component_name_inst = f"{component_name}_0"
        depth_debug_info = self.verilog_parameters[f"{node_name}_IN_DEPTH"]
        width_debug_info = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
        size_debug_info = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
        key_debug_info = "[{}][{}]".format(
            self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"],
            self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"],
        )

        return f"""
{component_name} #(
.OUT_DEPTH({node_name}_IN_DEPTH), // = {depth_debug_info}
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

    def _emit_hs_wires_top(self, from_node, to_node, is_start=False, is_end=False):
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

        assert (
            from_type == to_type and from_type == "fixed"
        ), "Only fixed point is supported."
        data_cast = f"assign {to_name} = {from_name};"
        if from_type == "fixed" and to_type == "fixed" and from_prec != to_prec:
            in_width = self.verilog_parameters[f"{to_param}_WIDTH"]
            in_size = self.verilog_parameters[f"{to_param}_SIZE"]
            out_width = self.verilog_parameters[f"{from_param}_WIDTH"]
            out_size = self.verilog_parameters[f"{from_param}_SIZE"]
            debug_info_in = f"// [{in_width}][{in_size}]"
            debug_info_out = f"// [{out_width}][{out_size}]"
            data_cast = f"""// {data_cast}
fixed_cast #(
    .IN_SIZE({from_param}_SIZE),
    .IN_WIDTH({from_param}_WIDTH),
    .IN_FRAC_WIDTH({from_param}_FRAC_WIDTH),
    .OUT_WIDTH({to_param}_WIDTH),
    .OUT_FRAC_WIDTH({to_param}_FRAC_WIDTH)
) {from_name}_{to_name}_cast (
    .data_in ({from_name}), {debug_info_out}
    .data_out({to_name}) {debug_info_in}
);
"""

        cast_file = "cast/fixed_cast.sv"
        if cast_file not in self.dependence_files:
            rtl_dir = os.path.join(self.project_dir, "hardware", "rtl")
            cast_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "hardware", cast_file
                )
            )
            shutil.copy(cast_dir, rtl_dir)
            self.dependence_files.append(cast_file)

        return f"""
assign {from_name}_ready  = {to_name}_ready;
assign {to_name}_valid    = {from_name}_valid;
{data_cast}
"""

    def _emit_hs2bram_wires_top(self, from_node, to_node, is_start=False, is_end=False):
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
        assert (
            from_type == to_type and from_type == "fixed"
        ), "Only fixed point is supported."
        data_cast = f"assign {to_name}_q0 = {from_name};"
        cast_name = from_name
        if from_type == "fixed" and to_type == "fixed" and from_prec != to_prec:
            in_width = self.verilog_parameters[f"{to_param}_WIDTH"]
            in_size = self.verilog_parameters[f"{to_param}_SIZE"]
            out_width = self.verilog_parameters[f"{from_param}_WIDTH"]
            out_size = self.verilog_parameters[f"{from_param}_SIZE"]
            debug_info_in = f"// [{in_width}][{in_size}]"
            debug_info_out = f"// [{out_width}][{out_size}]"
            cast_name = f"{from_name}q0_cast"
            data_cast = f"""// {data_cast}
logic [{from_param}_WIDTH-1:0] {cast_name} [{from_param}_SIZE-1:0];
fixed_cast #(
    .IN_SIZE({from_param}_SIZE),
    .IN_WIDTH({from_param}_WIDTH),
    .IN_FRAC_WIDTH({from_param}_FRAC_WIDTH),
    .OUT_WIDTH({to_param}_WIDTH),
    .OUT_FRAC_WIDTH({to_param}_FRAC_WIDTH)
) {from_name}_{to_name}_cast (
    .data_in ({from_name}), {debug_info_out}
    .data_out({cast_name}) {debug_info_in}
);
"""

        cast_file = "common/ram_block.sv"
        if cast_file not in self.dependence_files:
            rtl_dir = os.path.join(self.project_dir, "hardware", "rtl")
            cast_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "hardware", cast_file
                )
            )
            shutil.copy(cast_dir, rtl_dir)
            self.dependence_files.append(cast_file)
        cast_file = "cast/hs2bram_cast.sv"
        if cast_file not in self.dependence_files:
            rtl_dir = os.path.join(self.project_dir, "hardware", "rtl")
            cast_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "hardware", cast_file
                )
            )
            shutil.copy(cast_dir, rtl_dir)
            self.dependence_files.append(cast_file)

        to_node_name = vf(to_node.name)
        from_node_name = vf(from_node.name)
        size = self.verilog_parameters[f"{to_node_name}_IN_SIZE"]
        width = self.verilog_parameters[f"{to_node_name}_IN_WIDTH"]
        if is_start:
            in_size = size
        else:
            in_size = self.verilog_parameters[f"{from_node_name}_OUT_SIZE"]
        a_width = math.ceil(math.log2(size))
        return f"""
{data_cast}
hs2bram_cast #(
.IN_SIZE({from_param}_SIZE), // = {in_size}
.IN_WIDTH({to_param}_WIDTH), // = {width}
.ADDR_RANGE({to_param}_SIZE), // = {size}
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
.out_done({to_node_name}_done),
.clk(clk),
.rst(rst)
);
"""

    def _emit_bram2hs_wires_top(self, from_node, to_node, is_start=False, is_end=False):
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
        assert (
            from_type == to_type and from_type == "fixed"
        ), "Only fixed point is supported."
        data_cast = f"assign {to_name} = {from_name}_d0;"
        cast_name = f"{from_name}_d0"
        if from_type == "fixed" and to_type == "fixed" and from_prec != to_prec:
            in_width = self.verilog_parameters[f"{to_param}_WIDTH"]
            in_size = self.verilog_parameters[f"{to_param}_SIZE"]
            out_width = self.verilog_parameters[f"{from_param}_WIDTH"]
            out_size = self.verilog_parameters[f"{from_param}_SIZE"]
            debug_info_in = f"// [{in_width}][{in_size}]"
            debug_info_out = f"// [{out_width}][{out_size}]"
            cast_name = f"{from_name}_d0_cast"
            data_cast = f"""// {data_cast}
logic [{from_param}_WIDTH-1:0] {cast_name} [{from_param}_SIZE-1:0];
fixed_cast #(
    .IN_SIZE({from_param}_SIZE),
    .IN_WIDTH({from_param}_WIDTH),
    .IN_FRAC_WIDTH({from_param}_FRAC_WIDTH),
    .OUT_WIDTH({to_param}_WIDTH),
    .OUT_FRAC_WIDTH({to_param}_FRAC_WIDTH)
) {from_name}_{to_name}_cast (
    .data_in ({from_name}_d0), {debug_info_out}
    .data_out({cast_name}) {debug_info_in}
);
"""

        cast_file = "common/ram_block.sv"
        if cast_file not in self.dependence_files:
            rtl_dir = os.path.join(self.project_dir, "hardware", "rtl")
            cast_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "hardware", cast_file
                )
            )
            shutil.copy(cast_dir, rtl_dir)
            self.dependence_files.append(cast_file)
        cast_file = "cast/bram2hs_cast.sv"
        if cast_file not in self.dependence_files:
            rtl_dir = os.path.join(self.project_dir, "hardware", "rtl")
            cast_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "hardware", cast_file
                )
            )
            shutil.copy(cast_dir, rtl_dir)
            self.dependence_files.append(cast_file)

        to_node_name = vf(to_node.name)
        from_node_name = vf(from_node.name)
        size = self.verilog_parameters[f"{to_node_name}_IN_SIZE"]
        width = self.verilog_parameters[f"{to_node_name}_IN_WIDTH"]
        if is_start:
            in_size = size
        else:
            in_size = self.verilog_parameters[f"{from_node_name}_OUT_SIZE"]
        a_width = math.ceil(math.log2(size))
        return f"""
{data_cast}
bram2hs_cast #(
.OUT_SIZE({from_param}_SIZE), // = {in_size}
.OUT_WIDTH({to_param}_WIDTH), // = {width}
.ADDR_RANGE({to_param}_SIZE), // = {size}
.ADDR_WIDTH({a_width}) 
) {from_name}_{to_name}_hs2bram_cast (
.address0({from_name}_address0),
.ce0({from_name}_ce0),
.we0({from_name}_we0),
.d0({from_name}_d0),
.data_in_ready({to_name}_ready),
.data_in({cast_name}),
.data_in_valid({to_name}_valid),
.in_done({from_node_name}_done),
.clk(clk),
.rst(rst)
);
"""

    def _emit_bram_wires_top(self, from_node, to_node, is_start=False, is_end=False):
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
        assert (
            from_type == to_type and from_type == "fixed"
        ), "Only fixed point is supported."
        cast_name = f"{from_name}_d0"
        data_cast = ""
        if from_type == "fixed" and to_type == "fixed" and from_prec != to_prec:
            in_width = self.verilog_parameters[f"{to_param}_WIDTH"]
            in_size = self.verilog_parameters[f"{to_param}_SIZE"]
            out_width = self.verilog_parameters[f"{from_param}_WIDTH"]
            out_size = self.verilog_parameters[f"{from_param}_SIZE"]
            debug_info_in = f"// [{in_width}][{in_size}]"
            debug_info_out = f"// [{out_width}][{out_size}]"
            cast_name = f"{from_name}_d0_cast"
            data_cast = f"""// assign {cast_name}_q0 = {from_name}_d0
logic [{from_param}_WIDTH-1:0] {cast_name} [{from_param}_SIZE-1:0];
fixed_cast #(
    .IN_SIZE({from_param}_SIZE),
    .IN_WIDTH({from_param}_WIDTH),
    .IN_FRAC_WIDTH({from_param}_FRAC_WIDTH),
    .OUT_WIDTH({to_param}_WIDTH),
    .OUT_FRAC_WIDTH({to_param}_FRAC_WIDTH)
) {from_name}_{to_name}_cast (
    .data_in ({from_name}_d0), {debug_info_out}
    .data_out({cast_name}) {debug_info_in}
);
"""

        cast_file = "common/ram_block.sv"
        if cast_file not in self.dependence_files:
            rtl_dir = os.path.join(self.project_dir, "hardware", "rtl")
            cast_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "hardware", cast_file
                )
            )
            shutil.copy(cast_dir, rtl_dir)
            self.dependence_files.append(cast_file)
        cast_file = "cast/bram_cast.sv"
        if cast_file not in self.dependence_files:
            rtl_dir = os.path.join(self.project_dir, "hardware", "rtl")
            cast_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "hardware", cast_file
                )
            )
            shutil.copy(cast_dir, rtl_dir)
            self.dependence_files.append(cast_file)

        to_node_name = vf(to_node.name)
        from_node_name = vf(from_node.name)
        size = self.verilog_parameters[f"{to_node_name}_IN_SIZE"]
        width = self.verilog_parameters[f"{to_node_name}_IN_WIDTH"]
        a_width = math.ceil(math.log2(size))
        return f"""
{data_cast}
bram_cast #(
.IN_WIDTH({to_param}_WIDTH), // = {width}
.ADDR_RANGE({to_param}_SIZE), // = {size}
.ADDR_WIDTH({a_width}) 
) {from_name}_{to_name}_hs2bram_cast (
.address1({from_name}_address0),
.ce1({from_name}_ce0),
.we1({from_name}_we0),
.d1({from_name}_d0),
.in_done({from_node_name}_done),
.address0({to_name}_address0),
.ce0({to_name}_ce0),
.q0({cast_name}),
.out_start({to_node_name}_start),
.out_ready({to_node_name}_ready),
.out_done({to_node_name}_done),
.clk(clk),
.rst(rst)
);
"""

    def _emit_wires_top(self):
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

        nodes_in = self.nodes_in
        nodes_out = self.nodes_out
        node_in_name = vf(nodes_in[0].target)
        in_type = nodes_in[0].meta.parameters["common"]["args"]["data_in"]["type"]
        in_prec = nodes_in[0].meta.parameters["common"]["args"]["data_in"]["precision"]
        if nodes_in[0].meta.parameters["hardware"]["toolchain"] == "INTERNAL":
            wires += self._emit_hs_wires_top(nodes_in[0], nodes_in[0], is_start=True)
        elif nodes_in[0].meta.parameters["hardware"]["toolchain"] == "HLS":
            wires += self._emit_hs2bram_wires_top(
                nodes_in[0], nodes_in[0], is_start=True
            )
        else:
            assert False, "Unknown node toolchain for signal declarations."
        node_out_name = vf(nodes_out[0].target)
        out_type = nodes_out[0].meta.parameters["common"]["results"]["data_out"]["type"]
        out_prec = nodes_out[0].meta.parameters["common"]["results"]["data_out"][
            "precision"
        ]
        if nodes_out[0].meta.parameters["hardware"]["toolchain"] == "INTERNAL":
            wires += self._emit_hs_wires_top(nodes_out[0], nodes_out[0], is_end=True)
        elif nodes_out[0].meta.parameters["hardware"]["toolchain"] == "HLS":
            wires += self._emit_bram2hs_wires_top(
                nodes_out[0], nodes_out[0], is_end=True
            )
        else:
            assert False, "Unknown node toolchain for signal declarations."

        while nodes_in != nodes_out:
            next_nodes_in = []
            for node in nodes_in:
                node_name = vf(node.target)
                node_tc = node.meta.parameters["hardware"]["toolchain"]
                for next_node, x in node.users.items():
                    next_node_name = vf(next_node.target)
                    next_node_tc = next_node.meta.parameters["hardware"]["toolchain"]
                    if node_tc == "INTERNAL" and next_node_tc == "INTERNAL":
                        wires += self._emit_hs_wires_top(node, next_node)
                    elif node_tc == "HLS" and next_node_tc == "INTERNAL":
                        wires += self._emit_bram2hs_wires_top(node, next_node)
                    elif node_tc == "INTERNAL" and next_node_tc == "HLS":
                        wires += self._emit_hs2bram_wires_top(node, next_node)
                    elif node_tc == "HLS" and next_node_tc == "HLS":
                        wires += self._emit_bram_wires_top(node, next_node)
                    else:
                        assert False, "Unknown node toolchain for signal declarations."
                    if next_node.op == "output":
                        next_nodes_in.append(node)
                    else:
                        next_nodes_in.append(next_node)
            assert (
                nodes_in != next_nodes_in
            ), f"Parsing error: cannot find the next nodes: {nodes_in}."
            nodes_in = next_nodes_in
        return wires

    def emit_top(self):
        """
        Emit Verilog code for the top-level model
        """

        project_dir = self.project_dir
        rtl_dir = os.path.join(project_dir, "hardware", "rtl")

        top_file = os.path.join(rtl_dir, "{}.sv".format(self.project))
        top_design = open(top_file, "w")
        parameters_to_emit = self._emit_parameters_top()
        parameters_to_emit = _remove_last_comma(parameters_to_emit)
        interface_to_emit = self._emit_interface_top()
        interface_to_emit = _remove_last_comma(interface_to_emit)
        signals_to_emit = self._emit_siganls_top()
        components_to_emit = self._emit_components_top()
        wires_to_emit = self._emit_wires_top()
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
            self.project,
            time_to_emit,
            self.project,
            parameters_to_emit,
            interface_to_emit,
            signals_to_emit,
            components_to_emit,
            wires_to_emit,
        )
        top_design.write(module_inst)
        top_design.close()
        os.system(f"verible-verilog-format --inplace {top_file}")

    def _call_hls_flow(self, node, node_dir):
        """
        Synthesize the module using HLS
        """
        try:
            import torch_mlir
        except:
            print("TORCH-MLIR is not imported")

        if "torch_mlir" not in sys.modules:
            raise RuntimeError(f"TORCH_MLIR is required for synthesis.")

        # ----------------------------------
        #   Torch-MLIR
        # ----------------------------------
        x = torch.randn(node.meta.parameters["common"]["args"]["data_in"]["size"])
        module = torch_mlir.compile(
            node.meta.module, x, output_type="linalg-on-tensors"
        )
        mlir_dir = os.path.join(node_dir, f"{node.name}.linalg.mlir")
        with open(mlir_dir, "w", encoding="utf-8") as outf:
            outf.write(str(module))
        logger.debug(f"MLIR of module {node.name} successfully written into {mlir_dir}")
        assert os.path.isfile(mlir_dir), "Linalg MLIR generation failed."

        # ----------------------------------
        #   MLIR-Lowering
        # ----------------------------------
        lowered_dir = os.path.join(node_dir, f"{node.name}.affine.mlir")
        node_name = vf(node.name)
        # Lower Linalg MLIR to Affine MLIR
        cmd = [
            "mlir-opt",
            mlir_dir,
            "--linalg-bufferize",
            "--convert-linalg-to-affine-loops",
            "--canonicalize",
            "-o",
            lowered_dir,
        ]
        # if self.to_debug:
        #    cmd += ["--debug"]
        result = _execute(cmd, log_output=self.to_debug)
        assert os.path.isfile(lowered_dir), "Affine MLIR generation failed."
        logger.debug(
            f"MLIR Affine code of module {node.name} successfully written into {lowered_dir}"
        )

        mlir_dir = lowered_dir
        lowered_dir = os.path.join(node_dir, f"{node_name}.mase.mlir")
        hls_dir = os.path.join(node_dir, f"{node_name}.cpp")

        # Transform Affine MLIR for hardware generation and emit HLS code
        hls_param = _get_hls_parameters(node)
        cmd = [
            "mase-opt",
            mlir_dir,
            f"--preprocess-func=func-name={node_name}",
            "--canonicalize",
            f"--emit-hls=file-name={hls_dir} hls-param={hls_param}",
            "-o",
            lowered_dir,
        ]
        # if self.to_debug:
        #     cmd += ["--debug"]
        result = _execute(cmd, log_output=self.to_debug)
        assert os.path.isfile(hls_dir), "HLS code generation failed."
        logger.debug(
            f"HLS code of module {node.name} successfully written into {hls_dir}"
        )

        # Emit tcl for Vitis HLS
        hls_tcl = f"""
open_project -reset {node_name}
set_top {node_name}
add_files {node_name}.cpp
open_solution -reset "solution1"
set_part {self.target}
create_clock -period 4 -name default
config_bind -effort high
config_compile -pipeline_loops 1
csynth_design
# export_design -flow syn -rtl vhdl -format ip_catalog
"""
        hls_tcl_dir = os.path.join(node_dir, f"{node_name}.tcl")
        with open(hls_tcl_dir, "w", encoding="utf-8") as outf:
            outf.write(hls_tcl)
        logger.debug(
            f"HLS tcl of module {node.name} successfully written into {hls_tcl_dir}"
        )

        # Format HLS code so it is more readable
        cmd = [
            "clang-format",
            "-i",
            hls_dir,
        ]
        result = _execute(cmd, log_output=self.to_debug)

        # Call Vitis HLS for synthesis
        vitis_hls = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "scripts",
                "run-vitis-hls.sh",
            )
        )
        assert os.path.isfile(
            vitis_hls
        ), f"Vitis HLS not found. Please make sure if {vitis_hls} exists."
        cmd = [
            "bash",
            vitis_hls,
            hls_tcl_dir,
        ]
        result = _execute(cmd, log_output=self.to_debug, cwd=node_dir)

        if result:
            logger.error(f"Vitis HLS synthesis failed. {node.name}")
        else:
            logger.debug(
                f"Hardware of module {node.name} successfully generated by HLS"
            )
        return result

    def _emit_hls_component(self, node, queue):
        """
        Emit HLS component using MLIR
        """
        logger.debug(f"Synthesizing {node.name} using HLS")

        rtl_dir = os.path.join(self.project_dir, "hardware", "rtl")
        emit_parameters_in_rom_hls(node, rtl_dir)

        # Clean the HLS directory
        hls_dir = os.path.join(self.project_dir, "hardware", "hls")
        if not os.path.exists(hls_dir):
            os.mkdir(hls_dir)
        node_dir = os.path.join(hls_dir, node.name)
        if not os.path.exists(node_dir):
            os.mkdir(node_dir)
        for p in glob.glob(os.path.join(node_dir, "*")):
            if os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p)

        result = self._call_hls_flow(node, node_dir)
        queue.put(result)

    def _emit_hls_components(self, nodes):
        """
        Run HLS in parallel
        """
        hls_count = len(nodes)
        jobs = [None] * hls_count
        queue = Queue(hls_count)
        for i, node in enumerate(nodes):
            jobs[i] = Process(target=self._emit_hls_component, args=(node, queue))
            jobs[i].start()

        for job in jobs:
            job.join()

        err = 0
        for _ in range(hls_count):
            err += queue.get()
        if err:
            logger.error(f"HLS generation finished. {err} errors.")
        else:
            logger.info(f"HLS components generated. {err} errors.")
        assert not err

    def emit_components(self):
        """
        Emit the Verilog code of each module in the top-level model
        """

        rtl_dir = os.path.join(self.project_dir, "hardware", "rtl")
        hls_nodes = []
        for node in self.fx_graph.nodes:
            if node.op != "call_module" and node.op != "call_function":
                continue
            # If it is an internal module, just copy the files to the project
            if node.meta.parameters["hardware"]["toolchain"] == "INTERNAL":
                self.dependence_files = _add_dependence_files(
                    node.meta.parameters["hardware"]["dependence_files"],
                    self.dependence_files,
                )
                emit_parameters_in_rom_internal(node, rtl_dir)
            # If it is an HLS module, go through torch-mlir and mlir-hls flow
            # TODO: Call HLS synthesis processes in parallel
            elif node.meta.parameters["hardware"]["toolchain"] == "HLS":
                hls_nodes.append(node)
            else:
                raise NotImplementedError(f"EXTERNAL or HLS not supported yet.")

        self._emit_hls_components(hls_nodes)

        relative_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "hardware")
        )
        for svfile in self.dependence_files:
            shutil.copy(os.path.join(relative_dir, svfile), rtl_dir)
