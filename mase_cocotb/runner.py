from os import path, getenv
from shutil import rmtree
from pathlib import Path
import re
import inspect
from typing import Any
import torch

from cocotb.runner import get_runner, get_results
from mase_components.deps import MASE_HW_DEPS


def mase_runner(
    module_param_list: list[dict[str, Any]] = [dict()],
    extra_build_args: list[str] = [],
    trace: bool = False,
    seed: int = None,
):
    assert type(module_param_list) == list, "Need to pass in a list of dicts!"

    # Get file which called this function
    # Should be of form components/<group>/test/<module>_tb.py
    test_filepath = inspect.stack()[1].filename
    matches = re.search(r"mase_components/(\w*)/test/(\w*)_tb\.py", test_filepath)
    assert matches != None, "Function only works when called from test"
    group, module = matches.groups()

    # Group path is components/<group>
    group_path = Path(test_filepath).parent.parent

    # Components path is components/
    comp_path = group_path.parent

    # Try to find RTL file:
    # components/<group>/rtl/<module>.py
    module_path = group_path.joinpath("rtl").joinpath(f"{module}.sv")
    assert path.exists(module_path), f"{module_path} does not exist."

    SIM = getenv("SIM", "verilator")

    build_dir = group_path.joinpath(f"test/build/{module}")
    print(build_dir)
    if path.exists(build_dir):
        rmtree(build_dir)

    deps = MASE_HW_DEPS[f"{group}/{module}"]

    total_tests = 0
    total_fail = 0

    for i, module_params in enumerate(module_param_list):
        print("##########################################")
        print(f"#### TEST {i} : {module_params}")
        print("##########################################")
        test_work_dir = group_path.joinpath(f"test/build/{module}/test_{i}")
        runner = get_runner(SIM)
        runner.build(
            verilog_sources=[module_path],
            includes=[str(comp_path.joinpath(f"{d}/rtl/")) for d in deps],
            hdl_toplevel=module,
            build_args=[
                "--Wall",
                # Turn on assertions
                "--assert",
                # Verilator linter is overly strict.
                # Too many errors
                # These errors are in later versions of verilator
                "-Wno-GENUNNAMED",
                "-Wno-WIDTHEXPAND",
                "-Wno-WIDTHTRUNC",
                # Simulation Optimisation
                "-Wno-UNOPTFLAT",
                # Signal trace in dump.fst
                *(["--trace-fst", "--trace-structs"] if trace else []),
                *extra_build_args,
            ],
            parameters=module_params,
            build_dir=test_work_dir,
        )
        runner.test(
            hdl_toplevel=module,
            test_module=module + "_tb",
            seed=seed,
            results_xml="results.xml",
        )
        num_tests, fail = get_results(test_work_dir.joinpath("results.xml"))
        total_tests += num_tests
        total_fail += fail

    print("TEST RESULTS")
    print("    PASSED:", total_tests - total_fail)
    print("    FAILED:", total_fail)
    print("    NUM TESTS:", total_tests)

    return total_fail
