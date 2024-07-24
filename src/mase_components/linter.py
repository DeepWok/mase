import os, sys
from pathlib import Path
import subprocess

from chop.tools import get_logger, set_logging_verbosity
import mase_components
from mase_components.deps import MASE_HW_DEPS

logger = get_logger(f"linter")

COMPONENTS_PATH = Path(__file__).parents[0]


def run_lint(group):
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
        logger.info(f"Linting {file_path}")
        logger.info(f"----------------------------")

        module_path = f"{group}/{rtl_file.replace('.sv', '')}"

        if module_path not in MASE_HW_DEPS.keys():
            logger.warning(
                f"Module {module_path} is not included in dependencies file."
            )

        # * List include files
        include_files = [
            f"-I{COMPONENTS_PATH / group / 'rtl'}"
            for group in mase_components.get_modules()
        ]

        # * Run lint
        cmd = [
            "verilator",
            "--lint-only",
            "--Wall",
            "-Wno-GENUNNAMED",
            "-Wno-WIDTHEXPAND",
            "-Wno-WIDTHTRUNC",
            "-Wno-UNOPTFLAT",
            "-Wno-PINCONNECTEMPTY",
            "-Wno-UNUSEDSIGNAL",
            "-Wno-UNUSEDPARAM",
            file_path,
        ] + include_files

        logger.info(f"Executing {cmd}")

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
        logger.error(f"Failed linting on the following files: {failures}")
        sys.exit(1)
