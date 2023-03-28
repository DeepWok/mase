import glob
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
from ..graph.utils import get_module_by_name, vf
from .mase_rom_emitter import emit_node_parameters_in_rom

logger = logging.getLogger(__name__)


def _remove_last_comma(string):
    return string[0 : string.rfind(",")]


def _add_dependence_files(files, file_list):
    file_list.extend(f for f in files if f not in file_list)
    return file_list


def _execute(cmd, log_output: bool = True, log_file=None, cwd="."):
    if log_output:
        logger.debug(subprocess.list2cmdline(cmd))
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


class MaseVerilogEmitter(MaseGraph):
    def __init__(
        self,
        model=None,
        mode="auto",
        project_dir=".",
        project="top",
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
        self.project = project
        self.project_dir = os.path.join(project_dir, project)
        self.target = target
        self.num_targets = num_targets
        self.to_debug = to_debug
        self._init_project()
        self.verilog_parameters = {}

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
        self.verify()
        project_dir = self.project_dir
        rtl_dir = os.path.join(project_dir, "hardware", "rtl")
        if not os.path.exists(rtl_dir):
            os.mkdir(rtl_dir)
        self.emit_components()
        self.emit_top()

        files = ""
        for file in glob.glob(os.path.join(rtl_dir, "*.sv")):
            files += " " + file
        # os.system(f"verilator --lint-only {files}")

    def _emit_parameters(self):
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

    def _emit_interface(self):
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
                cap_key = key.upper()
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
                cap_key = key.upper()
                width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
                size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
                debug_info = f"// [{width}][{size}]"
                interface += f"""{debug_info}
output [{node_name}_{cap_key}_WIDTH-1:0] {node_name}_{key} [{node_name}_{cap_key}_SIZE-1:0],
output {node_name}_{key}_valid,
input  {node_name}_{key}_ready,
"""
        return interface

    def _emit_signals(self):
        """
        Emit internal signal declarations for the top-level module
        """

        signals = ""
        for node in self.fx_graph.nodes:
            if node.op != "call_module" and node.op != "call_function":
                continue
            node_name = vf(node.name)
            in_width = self.verilog_parameters[f"{node_name}_IN_WIDTH"]
            in_size = self.verilog_parameters[f"{node_name}_IN_SIZE"]
            out_width = self.verilog_parameters[f"{node_name}_OUT_WIDTH"]
            out_size = self.verilog_parameters[f"{node_name}_OUT_SIZE"]
            debug_info_in = f"// [{in_width}][{in_size}]"
            debug_info_out = f"// [{out_width}][{out_size}]"
            signals += f"""{debug_info_in}
logic [{node_name}_IN_WIDTH-1:0]  {node_name}_data_in        [{node_name}_IN_SIZE-1:0];
logic                             {node_name}_data_in_valid;
logic                             {node_name}_data_in_ready;
{debug_info_out}
logic [{node_name}_OUT_WIDTH-1:0] {node_name}_data_out            [{node_name}_OUT_SIZE-1:0];
logic                             {node_name}_data_out_valid;
logic                             {node_name}_data_out_ready;
"""
            for key, value in node.meta.parameters["common"]["args"].items():
                if key == "data_in":
                    continue
                # No top-level interface if the memory is stored on chip
                if (
                    node.meta.parameters["hardware"]["interface_parameters"][key][
                        "storage"
                    ]
                    != "BRAM"
                ):
                    continue
                cap_key = key.upper()
                width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
                size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
                debug_info = f"// [{width}][{size}]"
                signals += f"""{debug_info}
logic [{node_name}_{cap_key}_WIDTH-1:0]  {node_name}_{key}        [{node_name}_{cap_key}_SIZE-1:0];
logic                             {node_name}_{key}_valid;
logic                             {node_name}_{key}_ready;
{debug_info_out}
"""

            for key, value in node.meta.parameters["common"]["results"].items():
                if key == "data_out":
                    continue
                # No top-level interface if the memory is stored on chip
                if (
                    node.meta.parameters["hardware"]["interface_parameters"][key][
                        "storage"
                    ]
                    != "BRAM"
                ):
                    continue
                cap_key = key.upper()
                width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
                size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
                debug_info = f"// [{width}][{size}]"
                signals += f"""{debug_info}
logic [{node_name}_{cap_key}_WIDTH-1:0]  {node_name}_{key}        [{node_name}_{cap_key}_SIZE-1:0];
logic                             {node_name}_{key}_valid;
logic                             {node_name}_{key}_ready;
{debug_info_out}
"""

        return signals

    def _emit_components(self):
        """
        Emit component declarations for the top-level module
        """

        components = ""
        for node in self.fx_graph.nodes:
            if node.op != "call_module" and node.op != "call_function":
                continue
            node_name = vf(node.name)
            parameters = ""
            if node.meta.parameters["hardware"]["toolchain"] == "INTERNAL":
                for key, value in node.meta.parameters["hardware"][
                    "verilog_parameters"
                ].items():
                    key_value = self.verilog_parameters[f"{node_name}_{key}"]
                    debug_info = f"// = {key_value}"
                    parameters += f".{key}({node_name}_{key}), {debug_info}\n"
            parameters = _remove_last_comma(parameters)
            node_layer = get_module_by_name(self.model, node.target)
            component_name = node.meta.parameters["hardware"]["module"]
            node_name = vf(node.name)
            signals = ""
            for key, value in node.meta.parameters["common"]["args"].items():
                cap_key = key.upper().replace("DATA_", "")
                width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
                size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
                debug_info = f"// [{width}][{size}]"
                signals += f"""
.{key}({node_name}_{key}), {debug_info}
.{key}_valid({node_name}_{key}_valid),
.{key}_ready({node_name}_{key}_ready),
"""
            for key, value in node.meta.parameters["common"]["results"].items():
                cap_key = key.upper().replace("DATA_", "")
                width = self.verilog_parameters[f"{node_name}_{cap_key}_WIDTH"]
                size = self.verilog_parameters[f"{node_name}_{cap_key}_SIZE"]
                debug_info = f"// [{width}][{size}]"
                signals += f"""
.{key}({node_name}_{key}), {debug_info}
.{key}_valid({node_name}_{key}_valid),
.{key}_ready({node_name}_{key}_ready),
"""
            signals = _remove_last_comma(signals)
            component = """
{} #(
{}
) {} (
.clk(clk),
.rst(rst),
{}
);
""".format(
                component_name, parameters, node_name, signals
            )
            components += component

            for key, value in node.meta.parameters["common"]["args"].items():
                if key == "data_in":
                    continue
                # No top-level interface if the memory is stored on chip
                if (
                    node.meta.parameters["hardware"]["interface_parameters"][key][
                        "storage"
                    ]
                    != "BRAM"
                ):
                    continue
                if node.meta.parameters["hardware"]["toolchain"] == "INTERNAL":
                    components += self._emit_parameters_internal(key, value, node)
                elif node.meta.parameters["hardware"]["toolchain"] == "HLS":
                    components += self._emit_parameters_hls(key, value, node)
                else:
                    assert (
                        False
                    ), "Emitting external component input parameters is not supported."

            for key, value in node.meta.parameters["common"]["results"].items():
                if key == "data_out":
                    continue
                # No top-level interface if the memory is stored on chip
                if (
                    node.meta.parameters["hardware"]["interface_parameters"][key][
                        "storage"
                    ]
                    != "BRAM"
                ):
                    continue
                if node.meta.parameters["hardware"]["toolchain"] == "INTERNAL":
                    components += self._emit_parameters_internal(key, value, node)
                elif node.meta.parameters["hardware"]["toolchain"] == "HLS":
                    components += self._emit_parameters_hls(key, value, node)
                else:
                    assert (
                        False
                    ), "Emitting external component output parameters is not supported."

        return components

    def _emit_parameters_hls(self, key, value, node):
        return ""

    def _emit_parameters_internal(self, key, value, node):
        node_name = vf(node.name)
        cap_key = key.upper()
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

    def _emit_wires(self):
        """
        Emit internal signal connections for the top-level module
        """

        nodes_in = self.nodes_in
        nodes_out = self.nodes_out
        node_in_name = vf(nodes_in[0].target)
        node_out_name = vf(nodes_out[0].target)
        wires = f"""
assign data_in_ready  = {node_in_name}_data_in_ready;
assign {node_in_name}_data_in    = data_in;
assign {node_in_name}_data_in_valid = data_in_valid;
assign {node_out_name}_data_out_ready  = data_out_ready;
assign data_out    = {node_out_name}_data_out;
assign data_out_valid = {node_out_name}_data_out_valid;
"""
        while nodes_in != nodes_out:
            next_nodes_in = []
            for node in nodes_in:
                node_name = vf(node.target)
                for next_node, x in node.users.items():
                    next_node_name = vf(next_node.target)
                    if next_node.op == "call_module" or next_node.op == "call_function":
                        in_width = self.verilog_parameters[f"{next_node_name}_IN_WIDTH"]
                        in_size = self.verilog_parameters[f"{next_node_name}_IN_SIZE"]
                        out_width = self.verilog_parameters[f"{node_name}_OUT_WIDTH"]
                        out_size = self.verilog_parameters[f"{node_name}_OUT_SIZE"]
                        debug_info_in = ""
                        debug_info_out = ""
                        if self.to_debug:
                            debug_info_in = f"// [{in_width}][{in_size}]"
                            debug_info_out = f"// [{out_width}][{out_size}]"

                        wires += f"""
assign {node_name}_data_out_ready  = {next_node_name}_data_in_ready;
assign {next_node_name}_data_in_valid = {node_name}_data_out_valid;
// assign {next_node_name}_data_in = {node_name}_data_out;
fixed_cast #(
    .IN_SIZE({node_name}_OUT_SIZE),
    .IN_WIDTH({node_name}_OUT_WIDTH),
    .IN_FRAC_WIDTH({node_name}_OUT_FRAC_WIDTH),
    .OUT_WIDTH({next_node_name}_IN_WIDTH),
    .OUT_FRAC_WIDTH({next_node_name}_IN_FRAC_WIDTH)
) {node_name}_data_out_cast (
    .data_in ({node_name}_data_out), {debug_info_out}
    .data_out({next_node_name}_data_in) {debug_info_in}
);
"""
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
        parameters_to_emit = self._emit_parameters()
        parameters_to_emit = _remove_last_comma(parameters_to_emit)
        interface_to_emit = self._emit_interface()
        interface_to_emit = _remove_last_comma(interface_to_emit)
        signals_to_emit = self._emit_signals()
        components_to_emit = self._emit_components()
        wires_to_emit = self._emit_wires()
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

    def _emit_components_using_hls(self, node, node_dir):
        """
        Synthesize the module using HLS
        """
        try:
            import torch_mlir
        except:
            print("TORCH-MLIR is not imported")

        if "torch_mlir" not in sys.modules:
            raise RuntimeError(f"TORCH_MLIR is required for synthesis.")

        # Clear the folder
        for p in glob.glob(os.path.join(node_dir, "*")):
            if os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p)

        x = torch.randn(node.meta.parameters["common"]["args"]["data_in"]["size"])
        module = torch_mlir.compile(
            node.meta.module, x, output_type="linalg-on-tensors"
        )
        mlir_dir = os.path.join(node_dir, f"{node.name}.linalg.mlir")
        with open(mlir_dir, "w", encoding="utf-8") as outf:
            outf.write(str(module))
        logger.debug(f"MLIR of module {node.name} successfully written into {mlir_dir}")
        assert os.path.isfile(mlir_dir), "Linalg MLIR generation failed."

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

        # Get all the parameters needed for HLS
        hls_param = _get_hls_parameters(node)

        # Transform Affine MLIR for hardware generation and emit HLS code
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
        cmd = [
            "vitis_hls",
            hls_tcl_dir,
        ]
        # result = _execute(cmd, log_output=self.to_debug, cwd=node_dir)
        logger.debug(f"Hardware of module {node.name} successfully generated by HLS")

    def emit_components(self):
        """
        Emit the Verilog code of each module in the top-level model
        """

        project_dir = self.project_dir
        rtl_dir = os.path.join(project_dir, "hardware", "rtl")
        dependence_files = []
        for node in self.fx_graph.nodes:
            if node.op != "call_module" and node.op != "call_function":
                continue
            # If it is an internal module, just copy the files to the project
            if node.meta.parameters["hardware"]["toolchain"] == "INTERNAL":
                dependence_files = _add_dependence_files(
                    node.meta.parameters["hardware"]["dependence_files"],
                    dependence_files,
                )
                emit_node_parameters_in_rom(node, rtl_dir)
            # If it is an HLS module, go through torch-mlir and mlir-hls flow
            # TODO: Call HLS synthesis processes in parallel
            elif node.meta.parameters["hardware"]["toolchain"] == "HLS":
                logger.debug(f"Synthesizing {node.name} using HLS")
                hls_dir = os.path.join(project_dir, "hardware", "hls")
                if not os.path.exists(hls_dir):
                    os.mkdir(hls_dir)
                node_dir = os.path.join(hls_dir, node.name)
                if not os.path.exists(node_dir):
                    os.mkdir(node_dir)
                self._emit_components_using_hls(node, node_dir)
            else:
                raise NotImplementedError(f"EXTERNAL or HLS not supported yet.")
        relative_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "hardware")
        )
        for svfile in dependence_files:
            shutil.copy(os.path.join(relative_dir, svfile), rtl_dir)
