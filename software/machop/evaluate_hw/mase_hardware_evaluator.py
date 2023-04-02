import os
import sys
import glob
import logging

from ..graph.utils import vf
from ..synthesize.mase_verilog_emitter import _execute

logger = logging.getLogger(__name__)


def get_synthesis_results(model, mase_graph, target, output_dir):
    # TODO : may synthesize different partitions for seperate targets
    project_name = "synth_project"
    project_dir = os.path.join(output_dir, "hardware", project_name)
    tcl_dir = os.path.join(output_dir, "hardware", "syn.tcl")

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
        if node.meta.parameters["hardware"]["toolchain"] == "HLS":
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
    tcl_buff += f"""
set_property top {model} [current_fileset]
update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 20 
wait_on_run synth_1
open_run synth_1 -name synth_1
report_utilization -hierarchical -file {resource_report}
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets -name timing_1 -file {timing_report}
"""

    with open(tcl_dir, "w", encoding="utf-8") as outf:
        outf.write(tcl_buff)
    assert os.path.isfile(
        tcl_dir
    ), f"Vivado tcl not found. Please make sure if {tcl_dir} exists."

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
    logger.debug(cmd)
    result = _execute(cmd, log_output=True)
    assert not result, f"Vivado synthesis failed. {model}"
    logger.info(f"Hardware of module {model} successfully synthesized.")
