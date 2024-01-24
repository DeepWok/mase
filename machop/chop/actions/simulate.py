from os import getenv, PathLike
import torch
from cocotb.runner import get_runner, get_results
from pathlib import Path
import mase_components
from mase_components import get_modules

from .emit import emit


def simulate(
    model: torch.nn.Module,
    model_info,
    task: str,
    dataset_info,
    data_module,
    load_name: PathLike = None,
    load_type: str = None,
    run_emit: bool = False,
    skip_build: bool = False,
    skip_test: bool = False,
):
    SIM = getenv("SIM", "verilator")
    runner = get_runner(SIM)

    project_dir = Path.home() / ".mase" / "top"

    if run_emit:
        emit(model, model_info, task, dataset_info, data_module, load_name, load_type)

    if not skip_build:
        # To do: extract from mz checkpoint
        sources = [
            project_dir / "hardware" / "rtl" / "top.sv",
        ]

        runner.build(
            verilog_sources=sources,
            includes=[
                project_dir / "hardware" / "rtl",
            ]
            # Include all mase components
            + [
                Path(mase_components.__file__).parent / module / "rtl"
                for module in get_modules()
            ],
            hdl_toplevel="top",
            build_args=["-Wno-fatal", "-Wno-lint", "-Wno-style", "--trace"],
            parameters=[],  # use default parameters,
        )

    if not skip_test:
        # Add tb file to python path
        import sys

        sys.path.append(str(project_dir / "hardware" / "test"))

        runner.test(
            hdl_toplevel="top", test_module="mase_top_tb", hdl_toplevel_lang="verilog"
        )
    #     num_tests, fail = get_results("build/results.xml")
    # return num_tests, fail
