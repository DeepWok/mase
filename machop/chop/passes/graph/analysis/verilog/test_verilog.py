import logging
import os, glob
from pathlib import Path

from .cocotb import VerificationCase
from cocotb.runner import get_runner

from chop.passes.graph.utils import vf
from mase_cocotb.random_test import RandomSource, RandomSink, check_results

logger = logging.getLogger(__name__)


def get_dut_parameters(graph):
    parameter_map = {}

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        node_name = vf(node.name)

        for key, value in (
            node.meta["mase"].parameters["hardware"]["verilog_param"].items()
        ):
            if not isinstance(value, (int, float, complex, bool)):
                value = '"' + value + '"'
            assert (
                f"{node_name}_{key}" not in parameter_map.keys()
            ), f"{node_name}_{key} already exists in the parameter map"
            parameter_map[f"{node_name}_{key}"] = value
    return parameter_map


def runner(mg, project_dir, top_name):
    sim = os.getenv("SIM", "verilator")

    # TODO: Grab internal verilog source only. Need to include HLS hardware as well.
    sv_srcs = []
    for v in glob.glob(os.path.join(project_dir, "hardware", "rtl", "*.sv")):
        sv_srcs.append(os.path.relpath(v, os.getcwd()))

    # TODO: make samples and iterations variable
    tb = VerificationCase(samples=1, iterations=1)

    # TODO: work out the num
    for node in mg.nodes_in:
        for arg, arg_info in node.meta["mase"].parameters["common"]["args"].items():
            setattr(
                tb,
                arg,
                RandomSource(
                    name=arg,
                    samples=tb.samples * tb.iterations,
                    num=12324,
                    max_stalls=0,
                ),
            )

    for node in mg.nodes_out:
        for result, result_info in (
            node.meta["mase"].parameters["common"]["results"].items()
        ):
            setattr(
                tb,
                result,
                RandomSink(
                    name=result,
                    samples=tb.samples * tb.iterations,
                    num=12324,
                    max_stalls=0,
                ),
            )

    p = get_dut_parameters(mg)
    # logger.debug(p)

    # set parameters
    extra_args = []
    for k, v in p.items():
        extra_args.append(f"-G{k}={v}")
    logger.debug(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=sv_srcs,
        hdl_toplevel=top_name,
        build_args=extra_args,
    )
    runner.test(hdl_toplevel=top_name, test_module=f"{top_name}_tb")


def test_verilog_analysis_pass(mg, pass_args={}):
    """Test the top-level hardware design using Cocotb

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

    logger.info(f"Running hardware simulation using Cocotb")

    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )
    top_name = pass_args["top_name"] if "top_name" in pass_args.keys() else "top"
    logger.info(f"Project path: {project_dir}")

    runner(mg, project_dir, top_name)
    return mg, {}
