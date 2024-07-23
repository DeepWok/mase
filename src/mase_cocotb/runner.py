from os import path, getenv
import logging
from shutil import rmtree
from pathlib import Path
from copy import deepcopy
import re
import inspect
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

import torch

import cocotb
from cocotb.runner import get_runner, get_results
from mase_components.deps import MASE_HW_DEPS

logger = logging.getLogger("mase_runner")
logger.setLevel("INFO")


def _single_test(
    i: int,  # id
    deps: list[str],
    module: str,
    module_params: dict,
    module_path: Path,
    comp_path: Path,
    test_work_dir: Path,
    extra_build_args: list[str] = [],
    seed: int = None,
    trace: bool = False,
    skip_build: bool = False,
):
    print("# ---------------------------------------")
    print(f"# Test {i}")
    print("# ---------------------------------------")
    print(f"# Parameters:")
    print(f"# - {'Test Index'}: {i}")
    for k, v in module_params.items():
        print(f"# - {k}: {v}")
    print("# ---------------------------------------")

    runner = get_runner(getenv("SIM", "verilator"))
    if not skip_build:
        runner.build(
            verilog_sources=[module_path],
            includes=[str(comp_path.joinpath(f"{d}/rtl/")) for d in deps],
            hdl_toplevel=module,
            build_args=[
                # Verilator linter is overly strict.
                # Too many errors
                # These errors are in later versions of verilator
                "-Wno-GENUNNAMED",
                "-Wno-WIDTHEXPAND",
                "-Wno-WIDTHTRUNC",
                # Simulation Optimisation
                "-Wno-UNOPTFLAT",
                "-prof-c",
                "--assert",
                "--stats",
                # Signal trace in dump.fst
                *(["--trace-fst", "--trace-structs"] if trace else []),
                "-O2",
                "-build-jobs",
                "8",
                "-Wno-fatal",
                "-Wno-lint",
                "-Wno-style",
                *extra_build_args,
            ],
            parameters=module_params,
            build_dir=test_work_dir,
        )
    try:
        runner.test(
            hdl_toplevel=module,
            hdl_toplevel_lang="verilog",
            test_module=module + "_tb",
            seed=seed,
            results_xml="results.xml",
            build_dir=test_work_dir,
        )
        num_tests, fail = get_results(test_work_dir.joinpath("results.xml"))
    except Exception as e:
        print(f"Error occured while running Verilator simulation: {e}")
        num_tests = fail = 1

    return {
        "num_tests": num_tests,
        "failed_tests": fail,
        "params": module_params,
    }


def mase_runner(
    module_param_list: list[dict[str, Any]] = [dict()],
    extra_build_args: list[str] = [],
    trace: bool = False,
    seed: int = None,
    jobs: int = 1,
    skip_build: bool = False,
):
    assert type(module_param_list) == list, "Need to pass in a list of dicts!"

    start_time = time()

    # Get file which called this function
    test_filepath = inspect.stack()[1].filename

    matches = re.search(r"mase_components/(\w*)/test/(\w*)_tb\.py", test_filepath)
    if matches is None:
        matches = re.search(
            r"mase_components/(\w*)/(\w*)/test/(\w*)_tb\.py", test_filepath
        )

    assert (
        matches != None
    ), "Did not find file that matches <module>_tb.py in the test folder!"

    # Should be of form components/<group>/test/<module>_tb.py
    if len(matches.groups()) == 2:
        group, module = matches.groups()

        # Group path is components/<group>
        group_path = Path(test_filepath).parent.parent

        # Components path is components/
        comp_path = group_path.parent

        # Try to find RTL file:
        # components/<group>/rtl/<module>.py
        module_path = group_path.joinpath("rtl").joinpath(f"{module}.sv")
        deps_key = f"{group}/{module}"

    # Should be of form components/<group>/<subgroup>/test/<module>_tb.py
    elif len(matches.groups()) == 3:
        group, sub_group, module = matches.groups()

        # Group path is components/<group>
        group_path = Path(test_filepath).parent.parent.parent

        # Components path is components/
        comp_path = group_path.parent

        # Try to find RTL file:
        # components/<group>/<subgroup>/rtl/<module>.py
        module_path = (
            group_path.joinpath(sub_group).joinpath("rtl").joinpath(f"{module}.sv")
        )
        deps_key = f"{group}/{sub_group}/{module}"
    else:
        raise ValueError(f"Unexpected directory structure: {test_filepath}")

    assert path.exists(module_path), f"{module_path} does not exist."

    deps = MASE_HW_DEPS[deps_key]

    total_tests = 0
    total_fail = 0
    passed_cfgs = []
    failed_cfgs = []

    # Single threaded run
    if jobs == 1:

        for i, module_params in enumerate(module_param_list):
            test_work_dir = group_path.joinpath(f"test/build/{module}/test_{i}")
            results = _single_test(
                i=i,
                deps=deps,
                module=module,
                module_params=module_params,
                module_path=module_path,
                comp_path=comp_path,
                test_work_dir=test_work_dir,
                extra_build_args=extra_build_args,
                seed=seed,
                trace=trace,
                skip_build=skip_build,
            )
            total_tests += results["num_tests"]
            total_fail += results["failed_tests"]
            if results["failed_tests"]:
                failed_cfgs.append((i, module_params))
            else:
                passed_cfgs.append((i, module_params))

    # Multi threaded run
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            # TODO: add timeout
            future_to_job_meta = {}
            for i, module_params in enumerate(module_param_list):
                test_work_dir = group_path.joinpath(f"test/build/{module}/test_{i}")
                future = executor.submit(
                    _single_test,
                    i=i,
                    deps=deps,
                    module=module,
                    module_params=module_params,
                    module_path=module_path,
                    comp_path=comp_path,
                    test_work_dir=test_work_dir,
                    extra_build_args=extra_build_args,
                    seed=seed,
                    trace=trace,
                )
                future_to_job_meta[future] = {
                    "id": i,
                    "params": deepcopy(module_params),
                }

            # Wait for futures to complete
            for future in as_completed(future_to_job_meta):
                meta = future_to_job_meta[future]
                id = meta["id"]
                params = meta["params"]
                try:
                    result = future.result()
                except Exception as exc:
                    print("Test %r generated an exception: %s" % (id, exc))
                else:
                    print("Test %r is done. Result: %s" % (id, result))
                    total_tests += result["num_tests"]
                    total_fail += result["failed_tests"]
                    if result["failed_tests"]:
                        failed_cfgs.append((id, params))
                    else:
                        passed_cfgs.append((id, params))

    print("# ---------------------------------------")
    print("# Test Results")
    print("# ---------------------------------------")
    print("# - Time elapsed: %.2f seconds" % (time() - start_time))
    print("# - Jobs: %d" % (jobs))
    print("# - Passed: %d" % (total_tests - total_fail))
    print("# - Failed: %d" % (total_fail))
    print("# - Total : %d" % (total_tests))
    print("# ---------------------------------------")

    if len(passed_cfgs):
        passed_cfgs = sorted(passed_cfgs, key=lambda t: t[0])
        print(f"# Passed Configs")
        print("# ---------------------------------------")
        for i, params in passed_cfgs:
            print(f"# - test_{i}: {params}")
        print("# ---------------------------------------")

    if len(failed_cfgs):
        failed_cfgs = sorted(failed_cfgs, key=lambda t: t[0])
        print(f"# Failed Configs")
        print("# ---------------------------------------")
        for i, params in failed_cfgs:
            print(f"# - test_{i}: {params}")
        print("# ---------------------------------------")

    return total_fail


def simulate_pass(
    project_dir: Path,
    module_params: dict[str, Any] = {},
    extra_build_args: list[str] = [],
    trace: bool = False,
):
    rtl_dir = project_dir / "hardware" / "rtl"
    sim_dir = project_dir / "hardware" / "sim"
    test_dir = project_dir / "hardware" / "test" / "mase_top_tb"

    SIM = getenv("SIM", "verilator")

    runner = get_runner(SIM)
    print("hi")
    runner.build(
        verilog_sources=[rtl_dir / "top.sv"],
        includes=[rtl_dir],
        hdl_toplevel="top",
        build_args=[
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
            "-prof-c",
            "--stats",
            "--assert",
            "-O2",
            "-build-jobs",
            "8",
            "-Wno-fatal",
            "-Wno-lint",
            "-Wno-style",
            *extra_build_args,
        ],
        parameters=module_params,
        build_dir=sim_dir,
    )
    runner.test(
        hdl_toplevel="top",
        test_module="test",
        results_xml="results.xml",
    )
