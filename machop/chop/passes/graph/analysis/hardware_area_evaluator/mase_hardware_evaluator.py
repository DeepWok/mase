import glob
import logging
import os
import sys

from ..board_config import fpga_board_info
from ..graph.utils import vf
from ..utils import execute_cli

logger = logging.getLogger(__name__)


def get_synthesis_results(model, mase_graph, target, output_dir):
    # TODO : may synthesize different partitions for seperate targets
    project_name = "synth_project"
    hw_dir = os.path.join(output_dir, "hardware")
    project_dir = os.path.join(hw_dir, project_name)
    tcl_dir = os.path.join(hw_dir, "syn.tcl")

    tcl_buff = f"""
create_project -force {project_name} {project_dir} -part {target}
"""
    for file in glob.glob(os.path.join(output_dir, "hardware", "rtl", "*.sv")):
        tcl_buff += "add_files -norecurse {" + file + "}\n"
    for file in glob.glob(os.path.join(output_dir, "hardware", "rtl", "*.v")):
        tcl_buff += "add_files -norecurse {" + file + "}\n"
    for file in glob.glob(os.path.join(output_dir, "hardware", "rtl", "*.vhd")):
        tcl_buff += "add_files -norecurse {" + file + "}\n"

    for node in mase_graph.fx_graph.nodes:
        if node.op != "call_module" and node.op != "call_function":
            continue
        if node.meta["mase"].parameters["hardware"]["toolchain"] == "HLS":
            node_name = vf(node.name)
            for file in glob.glob(
                os.path.join(
                    output_dir,
                    "hardware",
                    "hls",
                    node_name,
                    node_name,
                    "solution1",
                    "syn",
                    "verilog",
                    "*.v",
                )
            ):
                tcl_buff += "add_files -norecurse {" + file + "}\n"
            for file in glob.glob(
                os.path.join(
                    output_dir,
                    "hardware",
                    "hls",
                    node_name,
                    node_name,
                    "solution1",
                    "syn",
                    "verilog",
                    "*.tcl",
                )
            ):
                tcl_buff += "source -norecurse {" + file + "}\n"
    resource_report = os.path.join(project_dir, "utils.rpt")
    timing_report = os.path.join(project_dir, "timing.rpt")
    power_report = os.path.join(project_dir, "power.rpt")
    clk_dir = os.path.join(hw_dir, "clock.xdc")
    tcl_buff += f"""
add_files -fileset constrs_1 -norecurse {clk_dir}
set_property top {model} [current_fileset]
update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 20
wait_on_run synth_1
open_run synth_1 -name synth_1
report_utilization -hierarchical -file {resource_report}
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets -name timing_1 -file {timing_report}
report_power -file {power_report} -name {{power_1}}
"""

    with open(tcl_dir, "w", encoding="utf-8") as outf:
        outf.write(tcl_buff)
    assert os.path.isfile(
        tcl_dir
    ), f"Vivado tcl not found. Please make sure if {tcl_dir} exists."

    clk_buff = (
        "create_clock -add -name clk -period "
        + str(fpga_board_info[target]["CLK"])
        + " [get_ports {clk}];"
    )
    with open(clk_dir, "w", encoding="utf-8") as outf:
        outf.write(clk_buff)
    assert os.path.isfile(
        clk_dir
    ), f"Vivado clk constraint not found. Please make sure if {clk_dir} exists."

    # Call Vivado for synthesis
    vivado = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "scripts",
            "run-vivado.sh",
        )
    )
    assert os.path.isfile(
        vivado
    ), f"Vivado not found. Please make sure if {vivado} exists."
    cmd = [
        "bash",
        vivado,
        tcl_dir,
    ]
    result = execute_cli(cmd, log_output=True, cwd=hw_dir)
    if result:
        assert False, f"Vivado synthesis failed. {model}"
    else:
        logger.info(f"Hardware of module {model} successfully synthesized.")
