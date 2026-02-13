import logging
import os

from chop.tools.utils import execute_cli

logger = logging.getLogger(__name__)


def run_cosim_analysis_pass(graph, pass_args={}):
    """
    Call XSIM for simulation
    """
    logger.info("Running cosimulation...")
    project_dir = (
        pass_args["project_dir"] if "project_dir" in pass_args.keys() else "top"
    )
    xsim = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "..",
            "scripts",
            "run-xsim.sh",
        )
    )
    cmd = [
        "bash",
        xsim,
        f"top_tb",
    ]
    sim_dir = os.path.join(project_dir, "hardware", "sim", "prj")
    log = pass_args["log_file"] if "log_file" in pass_args.keys() else "simulation.log"
    result = execute_cli(cmd, cwd=sim_dir, log_file=log)

    # TODO: seems xsim always returns 0 - needs to check
    if result:
        logger.error(f"Co-simulation failed.")
    else:
        logger.debug(f"Co-simulation succeeded.")
    return graph
