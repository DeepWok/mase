import logging, torch
from pathlib import Path
from textwrap import indent

from chop.passes.graph.utils import init_project, get_node_by_name

logger = logging.getLogger(__name__)


def _generate_input_4d(
    arg: str,
    channels: int,
    total_dim0: int,
    total_dim1: int,
    parallel_dim0: int,
    parallel_dim1: int,
    width: int,
    frac_width: int,
):
    return f"""
{arg}_in = list()
for _ in range({channels} * batches):
    {arg}_in.extend(gen_random_matrix_input(
        {total_dim0},  # total_dim0
        {total_dim1},  # total_dim1
        {parallel_dim0},  # compute_dim0
        {parallel_dim1},  # compute_dim1
        {width},  # width
        {frac_width}  # frac_width
    ))
inputs["{arg}"] = {arg}_in
""".strip()


def _reconstruct_input_4d(
    arg: str,
    channels: int,
    total_dim0: int,
    total_dim1: int,
    parallel_dim0: int,
    parallel_dim1: int,
    width: int,
    frac_width: int,
):
    return f"""
# INPUT: {arg}
{arg}_driver = inputs["{arg}"]
{arg}_batched = batched({arg}_driver, {(total_dim0 // parallel_dim0) * (total_dim1 // parallel_dim1)})
{arg}_mat_list = [rebuild_matrix(b, {total_dim0}, {total_dim1}, {parallel_dim0}, \
{parallel_dim1}) for b in {arg}_batched]
{arg}_recon = torch.stack({arg}_mat_list).reshape(-1, {channels}, {total_dim1}, \
{total_dim0})
{arg}_recon = sign_extend_t({arg}_recon, {width}).to(dtype=torch.float32) / \
(2 ** {frac_width})
recon_dict["{arg}"] = {arg}_recon
""".strip()


def _process_output_4d(
    arg: str,
    total_dim0: int,
    total_dim1: int,
    parallel_dim0: int,
    parallel_dim1: int,
    width: int,
    frac_width: int,
):
    # TODO: Add signedness to output cast
    return f"""
# OUTPUT: {arg}
{arg}_flat = model_out.reshape(-1, {total_dim1}, {total_dim0})
{arg}_flat = integer_quantizer_for_hw({arg}_flat, {width}, {frac_width})
{arg}_monitor_exp = list()
for i in range({arg}_flat.shape[0]):
    {arg}_monitor_exp.extend(split_matrix({arg}_flat[i], {total_dim0}, {total_dim1}, \
{parallel_dim0}, {parallel_dim1}))
return {arg}_monitor_exp
""".strip()


def _make_stream_driver(data, valid, ready):
    return f"StreamDriver(dut.clk, dut.{data}, dut.{valid}, dut.{ready})"


def _make_stream_monitor(data, valid, ready, width, signed, error_bits):
    return (
        f"ErrorThresholdStreamMonitor(dut.clk, dut.{data}, dut.{valid}, "
        f"dut.{ready}, width={width}, signed={signed}, error_bits={error_bits})"
    )


def _emit_cocotb_tb_str(graph, tb_dir: Path):
    # Drivers and Monitors
    drivers = []
    for arg in graph.meta["mase"]["common"]["args"].keys():
        drivers.append(_make_stream_driver(arg, f"{arg}_valid", f"{arg}_ready"))

    monitors = []
    for res in graph.meta["mase"]["common"]["results"].keys():
        prec = graph.meta["mase"]["common"]["results"][res]["precision"]
        monitors.append(
            _make_stream_monitor(
                res,
                f"{res}_valid",
                f"{res}_ready",
                prec[0],
                True,
                error_bits=5,  # TODO: determine error bits
            )
        )

    # Save torch model
    model_path = tb_dir / "model.pth"
    torch.save(graph.model, model_path)

    # Inputs, Reconstructs, Output Processing
    inputs = []
    reconstructs = []

    # Input processing
    graph_inputs = graph.meta["mase"]["common"]["nodes_in"]
    node_args = graph.meta["mase"]["common"]["args"].items()
    for i, (arg, arg_info) in enumerate(node_args):
        node_name = graph_inputs[i].name
        shape = arg_info["shape"]
        precision = arg_info["precision"]
        parallelism = get_node_by_name(graph.fx_graph, node_name).meta["mase"][
            "hardware"
        ]["parallelism"]

        # Input shape: (N, C, H, W)
        if len(shape) == 4:
            inputs.append(
                _generate_input_4d(
                    arg=arg,
                    channels=shape[1],
                    total_dim0=shape[3],
                    total_dim1=shape[2],
                    parallel_dim0=parallelism[3],
                    parallel_dim1=parallelism[2],
                    width=precision[0],
                    frac_width=precision[1],
                )
            )
            reconstructs.append(
                _reconstruct_input_4d(
                    arg=arg,
                    channels=shape[1],
                    total_dim0=shape[3],
                    total_dim1=shape[2],
                    parallel_dim0=parallelism[3],
                    parallel_dim1=parallelism[2],
                    width=precision[0],
                    frac_width=precision[1],
                )
            )

        # Input shape: (N, C)
        elif len(shape) == 2:
            # Batch dimension always set to 1 in metadata
            # inputs[arg] = torch.rand(([batches] + arg_info["shape"][1:]))
            raise NotImplementedError()

        else:
            raise Exception(f"Not sure how to drive {len(shape)} dimensional input.")

    # Output processing
    graph_outputs = graph.meta["mase"]["common"]["nodes_out"]
    results = graph.meta["mase"]["common"]["results"].items()
    assert len(results) == 1, "Only supporting single output!"
    result_name, result_info = next(iter(results))

    node_name = graph_outputs[0].name
    shape = result_info["shape"]
    precision = result_info["precision"]
    parallelism = get_node_by_name(graph.fx_graph, node_name).meta["mase"]["hardware"][
        "parallelism"
    ]

    if len(shape) == 4:
        output_processing = _process_output_4d(
            arg=result_name,
            total_dim0=shape[3],
            total_dim1=shape[2],
            parallel_dim0=parallelism[3],
            parallel_dim1=parallelism[2],
            width=precision[0],
            frac_width=precision[1],
        )
    elif len(shape) == 2:
        raise NotImplementedError()
    else:
        raise Exception(f"Not sure how to process {len(shape)} dimensional input.")

    driver_text = indent("\n".join(drivers), " " * 12)
    monitor_text = indent("\n".join(monitors), " " * 12)
    inputs_text = indent("\n".join(inputs), " " * 8)
    reconstruct_text = indent("\n".join(reconstructs), " " * 8)
    output_processing_text = indent(output_processing, " " * 8)

    testbench_template = f"""
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    ErrorThresholdStreamMonitor,
)
from mase_cocotb.utils import batched, sign_extend_t
from mase_cocotb.matrix_tools import (
    gen_random_matrix_input,
    rebuild_matrix,
    split_matrix,
)

from chop.passes.graph.transforms.quantize.quantizers.quantizers_for_hw import (
    integer_quantizer_for_hw
)


class MaseGraphTB(Testbench):
    def __init__(self, dut):
        super().__init__(dut, dut.clk, dut.rst)

        # Instantiate drivers
        self.input_drivers.extend([
{driver_text}
        ])

        # Instantiate monitors
        self.output_monitors.extend([
{monitor_text}
        ])

        # Load model
        self.model = torch.load("{model_path}")

    def generate_inputs(self, batches):
        \"\"\"Generates unsigned integer lists for the cocotb drivers.\"\"\"
        inputs = dict()
{inputs_text}
        return inputs

    def reconstruct(self, inputs):
        \"\"\"Reconstructs float tensors from integer lists for software model.\"\"\"
        recon_dict = dict()
{reconstruct_text}
        return recon_dict

    def output_processing(self, model_out):
        \"\"\"Casts float tensors outputed by software model into unsigned integer
        lists for cocotb monitors.\"\"\"
{output_processing_text}

    def load_drivers(self, inputs):
        \"\"\"Load the corresponding index driver using inputs dict.\"\"\"
        for arg_idx, arg_in in enumerate(inputs.values()):
            self.input_drivers[arg_idx].load_driver(arg_in)

    def load_monitors(self, expectation):
        \"\"\"Load the expected output.\"\"\"
        self.output_monitors[0].load_monitor(expectation)

    def all_monitors_empty(self):
        \"\"\"Assert that all monitor expectation queues are drained.\"\"\"
        for mon in self.output_monitors:
            if not mon.exp_queue.empty():
                return False
        return True
""".strip()
    return testbench_template


def _emit_cocotb_test(graph, project_dir: Path, trace: bool):
    tb_dir = project_dir / "hardware" / "test" / "mase_top_tb"

    test_template = f"""
import torch
from pathlib import Path

import cocotb
from cocotb.triggers import Timer

from mase_cocotb.runner import simulate_pass
{_emit_cocotb_tb_str(graph, tb_dir)}


@cocotb.test()
async def test(dut):
    tb = MaseGraphTB(dut)
    await tb.initialize()
    in_tensors = tb.generate_inputs(batches=3)
    recon_in = tb.reconstruct(in_tensors)
    model_out = tb.model(*list(recon_in.values()))
    mon_exp = tb.output_processing(model_out)

    tb.load_drivers(in_tensors)
    tb.load_monitors(mon_exp)

    await Timer(1000, units="us")
    assert tb.all_monitors_empty()


if __name__ == "__main__":
    simulate_pass(Path("{project_dir}"), trace={trace})
"""

    with open(tb_dir / "test.py", "w") as f:
        f.write(test_template)


def emit_cocotb_transform_pass(graph, pass_args={}):
    """
    Emit test bench and related files for simulation

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)

    - pass_args
        - project_dir -> str : the directory of the project
        - trace -> bool : trace waves in the simulation
    """
    logger.info("Emitting testbench...")
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )
    init_project(project_dir)

    tb_dir = project_dir / "hardware" / "test" / "mase_top_tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    trace_waves = pass_args.get("trace", False)
    _emit_cocotb_test(graph, project_dir, trace_waves)

    return graph, None
