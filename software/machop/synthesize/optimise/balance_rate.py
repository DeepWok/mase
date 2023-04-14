import logging
import math

from ...board_config import fpga_board_info
from ...utils import get_factors

logger = logging.getLogger(__name__)


def get_in_size(target, node):
    assert target == "xcu250-figd2104-2L-e"
    phy_bw = 10.31284  # Gb/s
    # todo: 40Gb/s, 100Gb/s

    mac_head = 10  # Bytes
    ip_head = 20  # Bytes
    udp_head = 8  # Bytes
    max_packet_size = 576  # Bytes, reassembly buffer size
    max_packet_size -= ip_head
    max_packet_size -= udp_head

    clk = fpga_board_info[target]["CLK"]  # ns
    in_width = node.meta.parameters["common"]["args"]["data_in"]["precision"][0]
    total_size = math.prod(node.meta.parameters["common"]["args"]["data_in"]["size"])
    in_size_candidates = get_factors(total_size)

    packets_num = math.ceil(total_size * in_width / (max_packet_size * 8))
    data_in = (
        packets_num * (mac_head + ip_head + udp_head) * 8 + total_size * in_width
    )  # bits

    for in_size in reversed(in_size_candidates):
        in_depth = total_size / in_size
        rate_in = data_in / (in_depth * clk)  # Gb/s
        if rate_in < phy_bw:
            return in_size
    return 1


def balance_rate(verilog_emitter):
    """
    Balance the rate between the nodes using the input throughput
    Start from the first nodes and then propagate to the rest of the nodes for balancing throughput
    """
    logger.info("Balancing the rates between hardware nodes...")

    nodes_in = verilog_emitter.nodes_in
    nodes_out = verilog_emitter.nodes_out

    assert len(nodes_in) == 1
    assert len(nodes_out) == 1

    # input_size = {"IN_SIZE": get_in_size(verilog_emitter.target, nodes_in[0])}
    input_size = {"IN_SIZE": 1}

    while True:
        next_nodes_in = []
        for node in nodes_in:
            node.meta.update_hardware_parameters(parameters=input_size)
            for next_node, x in node.users.items():
                if next_node.op != "output":
                    next_nodes_in.append(next_node)
        if len(next_nodes_in) == 0:
            break
        input_size = None
        assert (
            nodes_in != next_nodes_in
        ), f"Parsing error: cannot find the next nodes: {nodes_in}."
        nodes_in = next_nodes_in

    return verilog_emitter
