import logging
from typing import Tuple, Dict
import math
import os
import time
from multiprocessing import Process, Queue

from chop.passes.graph.utils import vf, v2p, init_project

logger = logging.getLogger(__name__)


def get_test_parameters(mg):
    """
    Extract the verilog parameters from the mase graph for cocotb testing
    """
    return {}


def get_dummy_inputs(mg):
    """
    Fetch test inputs from dataset or create a random one
    """
    return {}


def run_software_test(mg, inputs):
    """
    Run software model on given inputs
    """
    return {}


def run_cocotb_test(mg, parameters, inputs):
    """
    Create a cocotb test case and use mase runner to run hardware simulation
    """
    return {}


def compare_results(r0, r1):
    return r0 == r1


def test_verilog_analysis_pass(graph, pass_args={}):
    """Use cocotb to test the model design in Verilog

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

    logger.info("Testing the model in Verilog...")

    project_dir = (
        pass_args["project_dir"] if "project_dir" in pass_args.keys() else "top"
    )
    top_name = pass_args["top_name"] if "top_name" in pass_args.keys() else "top"

    parameters = get_test_parameters(graph)
    inputs = get_dummy_inputs(graph)
    software_results = run_software_test(graph, inputs)
    hardware_results = run_cocotb_test(graph, parameters, inputs)

    compare_results(software_results, hardware_results)

    return graph, {}
