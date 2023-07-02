import logging
import math
import os
import time

import torch

from ...board_config import fpga_board_info
from ...evaluate_hw.mase_hardware_evaluator import get_synthesis_results
from ...evaluate_hw.parse_synth_rpt import evaluate_node_area
from ...graph.utils import get_module_by_name, vf

logger = logging.getLogger(__name__)


def partition_target(verilog_emitter):
    # Run Vivado synthesis - after initial verilog is emitted
    # get_synthesis_results(
    #    verilog_emitter.project,
    #    verilog_emitter,
    #    verilog_emitter.target,
    #    verilog_emitter.project_dir,
    # )

    report = os.path.join(
        verilog_emitter.project_dir, "hardware", "synth_project", "utils.rpt"
    )
    area_results = evaluate_node_area(verilog_emitter.fx_graph, report)
    logger.debug(area_results)
    modified = []
    current_partition = 0
    partition_lut = 0

    max_lut = (
        fpga_board_info[verilog_emitter.target]["UTIL"]
        * fpga_board_info[verilog_emitter.target]["LUT"]
    )

    nodes_in = verilog_emitter.nodes_in
    nodes_out = verilog_emitter.nodes_out
    assert len(nodes_in) == 1
    assert len(nodes_out) == 1

    while True:
        next_nodes_in = []
        for node in nodes_in:
            node_lut = area_results[node.name]["LUT"]
            assert (
                node_lut < max_lut
            ), "A single node too large to fit one partition! reduce parallelism?"
            if partition_lut + node_lut > max_lut:
                current_partition += 1
                partition_lut = 0
            partition_lut += node_lut
            partition = (verilog_emitter.target, current_partition)
            logger.debug(node.name + str(partition))
            node.meta["mase"].parameters["hardware"]["PARTITION"] = partition
            modified.append(node)
            for next_node, _ in node.users.items():
                all_inputs_modified = True
                for next_node_input in next_node.all_input_nodes:
                    if next_node_input not in modified:
                        all_inputs_modified = False
                        break
                    else:
                        assert (
                            next_node_input.meta.parameters["hardware"]["PARTITION"]
                            == partition
                        ), "Unexpected residual case"
                if all_inputs_modified:
                    if next_node.op != "output":
                        next_nodes_in.append(next_node)
        if len(next_nodes_in) == 0:
            break
        assert (
            nodes_in != next_nodes_in
        ), f"Parsing error: cannot find the next nodes: {nodes_in}."
        nodes_in = next_nodes_in

    return verilog_emitter
