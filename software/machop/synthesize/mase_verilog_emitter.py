import glob
import logging
import multiprocessing
import os
import shutil
import subprocess
import time
from multiprocessing import Process, Queue

import toml
import torch
import torch.fx
import torch_mlir

from ..graph.mase_graph import MaseGraph
from ..graph.mase_metadata import MaseMetadata
from ..graph.utils import get_module_by_name, vf

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


class MaseVerilogEmitter(MaseGraph):
    def __init__(
        self,
        model=None,
        project_path=".",
        project="top",
        to_debug=False,
        target="xc7z020clg484-1",
        num_targets=1,
    ):
        super().__init__(model=model)
        self.project = project
        self.project_path = os.path.join(project_path, project)
        self.target = target
        self.num_targets = num_targets
        self.fx_graph = None
        self.nodes_in = []
        self.nodes_out = []
        self.parse()
        self.to_debug = to_debug

    # ----------------------------------------------------------
    # Emit hardware code
    # ----------------------------------------------------------
    def emit_verilog(self):
        project_path = self.project_path
        if not os.path.exists(project_path):
            os.mkdir(project_path)
        rtl_path = os.path.join(project_path, "rtl")
        if not os.path.exists(rtl_path):
            os.mkdir(rtl_path)
        self.emit_components()
        self.emit_top()

        files = ""
        for file in glob.glob(os.path.join(rtl_path, "*.sv")):
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
            if node.meta.parameters["hardware"]["target"] == "INTERNAL":
                for key, value in node.meta.parameters["hardware"][
                    "verilog_parameters"
                ].items():
                    if not isinstance(value, (int, float, complex, bool)):
                        value = '"' + value + '"'
                    parameters += f"parameter {node_name}_{key} = {value},\n"
            elif node.meta.parameters["hardware"]["target"] == "EXTERNAL":
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
                cap_key = key.upper()
                interface += f"""
input  [{node_name}_{cap_key}_WIDTH-1:0] {node_name}_{key} [{node_name}_{cap_key}_SIZE-1:0],
input  {node_name}_{key}_valid,
output {node_name}_{key}_ready,
"""
            for key, value in node.meta.parameters["common"]["results"].items():
                if key == "data_out":
                    continue
                interface += f"""
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
            signals += f"""
logic [{node_name}_IN_WIDTH-1:0]  {node_name}_data_in        [{node_name}_IN_SIZE-1:0];
logic                             {node_name}_data_in_valid;
logic                             {node_name}_data_in_ready;
logic [{node_name}_OUT_WIDTH-1:0] {node_name}_data_out            [{node_name}_OUT_SIZE-1:0];
logic                             {node_name}_data_out_valid;
logic                             {node_name}_data_out_ready;
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
            if node.meta.parameters["hardware"]["target"] == "INTERNAL":
                for key, value in node.meta.parameters["hardware"][
                    "verilog_parameters"
                ].items():
                    parameters += f".{key}({node_name}_{key}),\n"
            parameters = _remove_last_comma(parameters)
            node_layer = get_module_by_name(self.model, node.target)
            component_name = node.meta.parameters["hardware"]["module"]
            node_name = vf(node.name)
            signals = ""
            for key, value in node.meta.parameters["common"]["args"].items():
                signals += f"""
.{key}({node_name}_{key}),
.{key}_valid({node_name}_{key}_valid),
.{key}_ready({node_name}_{key}_ready),
"""
            for key, value in node.meta.parameters["common"]["results"].items():
                signals += f"""
.{key}({node_name}_{key}),
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
        return components

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
assign {node_in_name}_data_in    = data_out;
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
                    next_node_name = vf(node.target)
                    if next_node.op == "call_module" or next_node.op == "call_function":
                        wires += f"""
assign {node_name}_data_out_ready  = {next_node_name}_data_in_ready;
assign {next_node_name}_data_in    = {node_name}_data_out;
assign {next_node_name}_data_out_valid = {node_name}_data_out_valid;
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

        project_path = self.project_path
        rtl_path = os.path.join(project_path, "rtl")

        top_file = os.path.join(rtl_path, "{}.sv".format(self.project))
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
input rsk,
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

    def _emit_components_using_hls(self, node, node_path):
        """
        Synthesize the module using HLS
        """

        # Clear the folder
        for p in glob.glob(os.path.join(node_path, "*")):
            if os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p)

        x = torch.randn(node.meta.parameters["common"]["args"]["data_in"]["size"])
        module = torch_mlir.compile(
            node.meta.module, x, output_type="linalg-on-tensors"
        )
        mlir_path = os.path.join(node_path, f"{node.name}.linalg.mlir")
        with open(mlir_path, "w", encoding="utf-8") as outf:
            outf.write(str(module))
        logger.debug(
            f"MLIR of module {node.name} successfully written into {mlir_path}"
        )
        assert os.path.isfile(mlir_path), "Linalg MLIR generation failed."

        lowered_path = os.path.join(node_path, f"{node.name}.affine.mlir")
        node_name = vf(node.name)
        # Lower Linalg MLIR to Affine MLIR
        cmd = [
            "mlir-opt",
            mlir_path,
            "--linalg-bufferize",
            "--convert-linalg-to-affine-loops",
            "--canonicalize",
            "-o",
            lowered_path,
        ]
        # if self.to_debug:
        #    cmd += ["--debug"]
        logger.debug(subprocess.list2cmdline(cmd))
        result = _execute(cmd, log_output=self.to_debug)
        assert os.path.isfile(lowered_path), "Affine MLIR generation failed."
        logger.debug(
            f"MLIR Affine code of module {node.name} successfully written into {lowered_path}"
        )

        mlir_path = lowered_path
        lowered_path = os.path.join(node_path, f"{node_name}.mase.mlir")
        hls_path = os.path.join(node_path, f"{node_name}.cpp")

        # Transform Affine MLIR for hardware generation and emit HLS code
        cmd = [
            "mase-opt",
            mlir_path,
            f"--preprocess-func=func-name={node_name}",
            "--canonicalize",
            f"--emit-hls=file-name={hls_path}",
            "-o",
            lowered_path,
        ]
        # if self.to_debug:
        #     cmd += ["--debug"]
        logger.debug(subprocess.list2cmdline(cmd))
        result = _execute(cmd, log_output=self.to_debug)
        assert os.path.isfile(hls_path), "HLS code generation failed."
        logger.debug(
            f"HLS code of module {node.name} successfully written into {hls_path}"
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
        hls_tcl_path = os.path.join(node_path, f"{node_name}.tcl")
        with open(hls_tcl_path, "w", encoding="utf-8") as outf:
            outf.write(hls_tcl)
        logger.debug(
            f"HLS tcl of module {node.name} successfully written into {hls_tcl_path}"
        )

        # Format HLS code so it is more readable
        cmd = [
            "clang-format",
            "-i",
            hls_path,
        ]
        logger.debug(subprocess.list2cmdline(cmd))
        result = _execute(cmd, log_output=self.to_debug)
        # Call Vitis HLS for synthesis
        cmd = [
            "vitis_hls",
            hls_tcl_path,
        ]
        logger.debug(subprocess.list2cmdline(cmd))
        result = _execute(cmd, log_output=self.to_debug, cwd=node_path)
        logger.debug(f"Hardware of module {node.name} successfully generated by HLS")

    def emit_components(self):
        """
        Emit the Verilog code of each module in the top-level model
        """

        project_path = self.project_path
        rtl_path = os.path.join(project_path, "rtl")
        dependence_files = []
        for node in self.fx_graph.nodes:
            if node.op != "call_module" and node.op != "call_function":
                continue
            # If it is an internal module, just copy the files to the project
            if node.meta.parameters["hardware"]["target"] == "INTERNAL":
                dependence_files = _add_dependence_files(
                    node.meta.parameters["hardware"]["dependence_files"],
                    dependence_files,
                )
            # If it is an HLS module, go through torch-mlir and mlir-hls flow
            # TODO: Call HLS synthesis processes in parallel
            elif node.meta.parameters["hardware"]["target"] == "HLS":
                logger.debug(f"Synthesizing {node.name} using HLS")
                hls_path = os.path.join(project_path, "hls")
                if not os.path.exists(hls_path):
                    os.mkdir(hls_path)
                node_path = os.path.join(hls_path, node.name)
                if not os.path.exists(node_path):
                    os.mkdir(node_path)
                self._emit_components_using_hls(node, node_path)
            else:
                raise NotImplementedError(f"EXTERNAL or HLS not supported yet.")
        relative_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "hardware")
        )
        for svfile in dependence_files:
            shutil.copy(os.path.join(relative_path, svfile), rtl_path)
