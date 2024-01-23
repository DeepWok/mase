from os import getenv
from cocotb.runner import get_runner, get_results
from pathlib import Path
import mase_components
from mase_components import get_modules


def simulate(skip_build: bool = False, skip_test: bool = False):
    SIM = getenv("SIM", "verilator")
    runner = get_runner(SIM)

    if not skip_build:
        # To do: extract from mz checkpoint
        project_dir = Path.home() / ".mase" / "top"
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
            build_args=[
                "-Wno-fatal",
                "-Wno-lint",
                "-Wno-style",
            ],
            parameters=[],  # use default parameters,
        )

    if not skip_test:
        runner.test(
            hdl_toplevel="top", test_module="top_tb", hdl_toplevel_lang="verilog"
        )
        num_tests, fail = get_results("build/results.xml")
    return num_tests, fail
