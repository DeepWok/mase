import os, glob
from chop.passes.graph.utils import vf

from .cocotb import VerificationCase
from mase_cocotb.random_test import RandomSource, RandomSink, check_results
from cocotb.runner import get_runner


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


def get_dependence_files(graph):
    f = []
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        f += node.meta["mase"].parameters["hardware"]["dependence_files"]

    f = list(dict.fromkeys(f))
    return f


def runner(mg):
    sim = os.getenv("SIM", "verilator")

    verilog_sources = get_dependence_files(mg)
    for i, v in enumerate(verilog_sources):
        verilog_sources[i] = os.path.relpath(
            os.path.join("/workspace", "mase_components", v), os.getcwd()
        )
    # TODO: make project name variable
    for v in glob.glob("./top/hardware/rtl/*.sv"):
        verilog_sources.append(os.path.relpath(v, os.getcwd()))

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
    print(p)

    # set parameters
    extra_args = []
    for k, v in p.items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="top",
        build_args=extra_args,
    )
    runner.test(hdl_toplevel="top", test_module="top_tb")


def test_verilog_analysis_pass(mg, pass_args={}):
    runner(mg)
    return mg, {}
