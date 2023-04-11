import os
import sys
import glob
import logging

from ..graph.utils import vf

logger = logging.getLogger(__name__)


def _parse_line_for_area(line):
    nums = line.replace(" ", "").split("|")
    return (
        int(nums[3]),
        int(nums[4]),
        int(nums[5]),
        int(nums[6]),
        int(nums[7]),
        int(nums[8]),
        int(nums[9]),
        int(nums[10]),
    )


def evaluate_node_area(fx_graph, report):
    """
    Parse synthesis report and return a dictionary of node area.
    """

    assert os.path.isfile(report), f"Cannot find the area report: {report}"
    with open(report, "r") as file:
        buff = file.read().split("\n")

    area = {}
    for node in fx_graph.nodes:
        node_name = vf(node.name)
        # In most cases, only LUT, FF, BRAM36 and DSP are needed
        area[node_name] = {
            "LUT": 0,
            "LLUT": 0,
            "LUTRAM": 0,
            "FF": 0,
            "RAM36": 0,
            "RAMB18": 0,
            "URAM": 0,
            "DSP": 0,
        }
        for line in buff:
            if line.startswith(f"|   {node_name}_"):
                lut, llut, lutram, ff, ram36, ram18, uram, dsp = _parse_line_for_area(
                    line
                )
                # += because a node may contain multiple components such as parameter sources
                area[node_name]["LUT"] += lut
                area[node_name]["LLUT"] += llut
                area[node_name]["LUTRAM"] += lutram
                area[node_name]["FF"] += ff
                area[node_name]["RAM36"] += ram36
                area[node_name]["RAMB18"] += ram18
                area[node_name]["URAM"] += uram
                area[node_name]["DSP"] += dsp
    return area


def evaluate_fmax(report):
    """
    Parse synthesis report and return a Fmax.
    """
    assert os.path.isfile(report), f"Cannot find the timing report: {report}"

    f = open(report, "r")
    for line in f:
        if "Design Timing Summary" in line:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            values = list(filter(lambda a: a != "", line.split(" ")))
            wns = float(values[0])
        if "Clock Summary" in line:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            values = list(filter(lambda a: a != "", line.split(" ")))
            clk = float(values[3])
            fmax = 1000 / (clk - wns)
            f.close()
            return "{:.2f}".format(fmax)
    f.close()
    return None
