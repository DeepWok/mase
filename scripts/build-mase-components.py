import mase_components
from mase_components.deps import MASE_HW_DEPS

import concurrent.futures
import subprocess

from pathlib import Path

import logging
from tabulate import tabulate
import emoji

import argparse

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s : %(levelname)s : %(message)s"
)

logger = logging.getLogger()

MASE_COMPONENTS_PATH = Path(mase_components.__file__).parents[0]
MASE_PATH = Path(mase_components.__file__).parents[2]

SOURCE_VIVADO = """
    source /mnt/applications/Xilinx/23.1/Vivado/2023.1/settings64.sh;
    source /mnt/applications/Xilinx/23.1/Vitis/2023.1/settings64.sh;
"""

MAX_THREADS = 16


def launch_build(group, module):
    logging.info(f"Creating build for {group}/{module}")
    cmd = f"""
        {SOURCE_VIVADO}
        cd {MASE_COMPONENTS_PATH}/{group};
        vivado -mode batch -source {MASE_PATH}/scripts/build-mase-module.tcl -log vivado-{group}-{module}.log -tclargs {group} {module}
        # rm -rf vivado*
    """
    try:
        subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True,
            check=True,
            executable="/bin/bash",
        )
        return f"{group}/{module}", None  # Success
    except subprocess.CalledProcessError as e:
        logger.error(f"Error while building {group}/{module}: {e}")
        return None, f"{group}/{module}"  # Failure


def log_dir(build):
    group, module = build.split("/")
    return f"{MASE_COMPONENTS_PATH}/{group}/vivado-{group}-{module}.log"


def pretty_summary(results):
    success_cnt = len(results["success"])
    failure_cnt = len(results["failure"])
    total_cnt = len(results["success"]) + len(results["failure"])
    logger.info(
        f"Build job finished. Success: {success_cnt}/{total_cnt}, Failure: {failure_cnt}/{total_cnt}"
    )

    # Inside the main function, where the build summary is generated
    table_headers = ["Status", "Build", "Log"]
    table_data = []

    # Successful builds
    success_emoji = emoji.emojize(":check_mark_button:")
    success_builds = [
        [success_emoji, build, log_dir(build)] for build in results["success"]
    ]
    table_data.extend(success_builds)

    # Failed builds
    failure_emoji = emoji.emojize(":cross_mark:")
    failure_builds = [
        [failure_emoji, build, log_dir(build)] for build in results["failure"]
    ]
    table_data.extend(failure_builds)

    # Logging the build summary in a formatted table
    logger.info(
        f"Build summary:\n{tabulate(table_data, headers=table_headers, tablefmt='fancy_grid')}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build MASE hardware components.")
    parser.add_argument(
        "-t",
        "--max-workers",
        type=int,
        default=16,
        help="Maximum number of workers for concurrent builds (default: 16)",
    )
    args = parser.parse_args()

    results = {"success": [], "failure": []}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        # Submit build tasks
        futures = {
            executor.submit(launch_build, *key.split("/")): key
            for key in MASE_HW_DEPS.keys()
        }

        # Process completed tasks
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            success, failure = future.result()
            if success:
                results["success"].append(success)
            elif failure:
                results["failure"].append(failure)

    pretty_summary(results)

    if len(results["failure"]) > 0:
        raise RuntimeError


if __name__ == "__main__":
    main()
