import os
import subprocess
from pathlib import Path

from chop.passes.graph.utils import init_project
from chop.tools import get_logger, set_logging_verbosity
import mase_components
from mase_components.deps import MASE_HW_DEPS

logger = get_logger(f"emit_vivado_project")

COMPONENTS_PATH = Path(mase_components.__file__).parents[0]


def generate_tcl_script(top_name, vivado_project_path, include_groups, project_dir):
    logger.info(
        f"Writing Vivado project generation script: {vivado_project_path}/build.tcl"
    )

    tcl_script_template = f"""
set_param board.repoPaths {{{str(Path.home())}/shared/board-files}}
create_project  {top_name}_build_project {vivado_project_path} -part xcu280-fsvh2892-2L-e
set_property board_part xilinx.com:au280:part0:1.1 [current_project]
"""
    for include_group in include_groups:
        tcl_script_template += f"""\nadd_files {include_group}"""

    tcl_script_template += f"\n\nset_property top top [current_fileset]"

    tcl_script_template += f"""
update_compile_order -fileset sources_1
"""

    # * Package IP
    tcl_script_template += f"""
ipx::package_project -root_dir {project_dir}/hardware/ip_repo -vendor user.org -library user -taxonomy /UserIP -import_files
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
set_property  ip_repo_paths  {project_dir}/hardware/ip_repo [current_project]
update_ip_catalog
"""

    with open(f"{vivado_project_path}/build.tcl", "w") as file:
        file.write(tcl_script_template)


def emit_vivado_project_transform_pass(graph, pass_args={}):
    """Emit the Vivado project containing the generated Verilog and all required IPs

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)


    - pass_args
        - project_dir -> str : the directory of the project for cosimulation
        - top_name -> str : top-level name
    """

    # * Check if Vivado is available by running the command
    try:
        subprocess.run(["vivado", "-version"], capture_output=True, text=True)
    except:
        logger.warning(
            "Vivado is not available, skipping emit_vivado_project_transform_pass."
        )
        return graph, {}

    logger.info("Emitting Vivado project...")

    # Create project directory, and the verilog is emmited to {project_name}/hardware/rtl
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )
    top_name = pass_args["top_name"] if "top_name" in pass_args.keys() else "top"
    init_project(project_dir)
    vivado_project_path = os.path.join(
        project_dir, "hardware", f"{top_name}_build_project"
    )
    os.makedirs(vivado_project_path, exist_ok=True)

    # * List include files
    include_groups = [
        f"{COMPONENTS_PATH / group / 'rtl'}"
        for group in mase_components.get_modules()
        if group != "vivado"
    ] + [project_dir / "hardware" / "rtl"]

    generate_tcl_script(top_name, vivado_project_path, include_groups, project_dir)

    logger.info(f"Emitting Vivado project at: {vivado_project_path}")
    cmd = [
        "vivado",
        "-mode",
        "batch",
        "-log",
        f"{vivado_project_path}/project_build.log",
        "-source",
        f"{vivado_project_path}/build.tcl",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    return graph, {}
