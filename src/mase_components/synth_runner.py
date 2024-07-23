import subprocess
from pathlib import Path
import os, sys

from chop.tools import get_logger, set_logging_verbosity
import mase_components
from mase_components.deps import MASE_HW_DEPS

logger = get_logger(f"linter")

COMPONENTS_PATH = Path(__file__).parents[0]


def generate_tcl_script(group, module_name, include_groups, synth_project_path):
    os.makedirs(synth_project_path, exist_ok=True)
    tcl_script_template = f"""
set_param board.repoPaths {{{str(Path.home())}/shared/board-files}}
create_project synth_project_{group}_{module_name} {synth_project_path} -part xcu280-fsvh2892-2L-e
set_property board_part xilinx.com:au280:part0:1.1 [current_project]
"""
    for include_group in include_groups:
        tcl_script_template += f"""\nadd_files {include_group}"""

    tcl_script_template += f"\n\nset_property top {module_name} [current_fileset]"

    tcl_script_template += """
update_compile_order -fileset sources_1
launch_runs synth_1
wait_on_runs synth_1
"""

    with open(f"{synth_project_path}/build.tcl", "w") as file:
        file.write(tcl_script_template)


def run_synth(group):
    comp_path = COMPONENTS_PATH / group / "rtl"
    rtl_files = [
        file
        for file in os.listdir(comp_path)
        if file.endswith(".sv") or file.endswith(".v")
    ]

    successes = []
    failures = []

    for rtl_file in rtl_files:
        file_path = comp_path / rtl_file
        logger.info(f"Synthesizing {file_path}")
        logger.info(f"----------------------------")

        module_name = rtl_file.replace(".sv", "")
        module_path = f"{group}/{module_name}"

        if module_path not in MASE_HW_DEPS.keys():
            logger.warning(
                f"Module {module_path} is not included in dependencies file."
            )

        # * List include files
        include_groups = [
            f"{COMPONENTS_PATH / group / 'rtl'}"
            for group in mase_components.get_modules()
            if group != "vivado"
        ]

        synth_project_path = (
            f"{COMPONENTS_PATH}/{group}/synth/synth_project_{group}_{module_name}"
        )

        logger.debug(f"Include files: {include_groups}")

        logger.info(f"Generating build TCL script for module: {module_path}")
        generate_tcl_script(group, module_name, include_groups, synth_project_path)

        logger.info(f"Launching Vivado synthesis for module: {module_path}")
        cmd = [
            "vivado",
            "-mode",
            "batch",
            "-log",
            f"{synth_project_path}/vivado.log",
            "-source",
            f"{synth_project_path}/build.tcl",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # * Process result
        if result.stderr == "":
            successes.append(rtl_file)
        else:
            logger.error(result.stderr)
            failures.append(rtl_file)

    # * Print summary
    logger.info(f"=========== SUMMARY ===========")
    logger.info(
        f"PASS: {len(successes)}/{len(rtl_files)}, FAIL: {len(failures)}/{len(rtl_files)}"
    )

    if len(failures) > 0:
        logger.error(f"Failed synthesizing the following modules: {failures}")
        sys.exit(1)
