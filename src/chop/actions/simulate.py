import sys
from os import getenv, PathLike

import torch
from pathlib import Path
import time
import warnings
from cocotb.runner import get_runner, get_results

from chop.tools import get_logger
import mase_components
from mase_components import get_modules
from .emit import emit

import glob, os

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Python runners and associated APIs are an experimental feature and subject to change.",
)
logger = get_logger(__name__)
logger.setLevel("DEBUG")


def simulate(
    model: torch.nn.Module = None,
    model_info=None,
    task: str = "",
    dataset_info=None,
    data_module=None,
    load_name: PathLike = None,
    load_type: str = None,
    run_emit: bool = False,
    skip_build: bool = False,
    skip_test: bool = False,
    trace_depth: int = 3,
    gui: bool = False,
    waves: bool = False,
    simulator: str = "verilator",
):
    SIM = getenv("SIM", simulator)
    runner = get_runner(SIM)

    project_dir = Path.home() / ".mase" / "top"

    if run_emit:
        emit(model, model_info, task, dataset_info, data_module, load_name, load_type)

    if not skip_build:
        # To do: extract from mz checkpoint
        if simulator == "questa":
            sources = glob.glob(os.path.join(project_dir / "hardware" / "rtl", "*.sv"))
            build_args = []

        elif simulator == "verilator":
            # sources = ["../../../top.sv"]
            sources = glob.glob(os.path.join(project_dir / "hardware" / "rtl", "*.sv"))
            build_args = [
                "-Wno-fatal",
                "-Wno-lint",
                "-Wno-style",
                "--trace-fst",
                "--trace-structs",
                "--trace-depth",
                str(trace_depth),
            ]
        else:
            raise ValueError(f"Unrecognized simulator: {simulator}")

        includes = [
            project_dir / "hardware" / "rtl",
        ] + [
            Path(mase_components.__file__).parent / module / "rtl"
            for module in get_modules()
        ]

        build_start = time.time()

        runner.build(
            verilog_sources=sources,
            includes=includes,
            hdl_toplevel="top",
            build_args=build_args,
            parameters=[],  # use default parameters,
        )

        build_end = time.time()
        logger.info(f"Build finished. Time taken: {build_end - build_start:.2f}s")

    if not skip_test:
        # Add tb file to python path

        sys.path.append(str(project_dir / "hardware" / "test"))

        test_start = time.time()
        runner.test(
            hdl_toplevel="top",
            test_module="mase_top_tb",
            hdl_toplevel_lang="verilog",
            gui=gui,
            waves=waves,
        )
        test_end = time.time()
        logger.info(f"Test finished. Time taken: {test_end - test_start:.2f}s")
    #     num_tests, fail = get_results("build/results.xml")
    # return num_tests, fail
