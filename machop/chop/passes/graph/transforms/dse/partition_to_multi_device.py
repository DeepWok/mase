import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

from chop.passes.graph.analysis.utils import get_hardware_nodes


def default_cluster_config(num_devices):
    cluster_config = {}
    for device_idx in range(num_devices):
        cluster_config[device_idx] = {
            "device_id": device_idx,
            "part": "xcu250-figd2104-2L-e",
            "nodes": [],
        }
    return cluster_config


def apply_naive_partition(mg, cluster_config=None, device_count=1):
    hw_nodes = get_hardware_nodes(mg)

    # TODO: Mode devices than hardware-mappable nodes
    if device_count > len(hw_nodes):
        random_devices = list(cluster_config.keys())
        np.random.shuffle(random_devices)
        for node_idx, node in enumerate(hw_nodes):
            device = random_devices[node_idx]
            node.meta["mase"].parameters["hardware"]["device_id"] = device
            cluster_config[device] = node

    # TODO: More nodes than devices, or same amount
    else:  # device_count <= len(hw_nodes)
        """
        Generate random partitioning of elements in objects, preserving
            the order of elements within each partition
        """

        cut_points = np.arange(1, len(hw_nodes))
        np.random.shuffle(cut_points)
        cut_points = cut_points[: device_count - 1]
        cut_points = np.sort(cut_points)
        hw_nodes = np.split(hw_nodes, cut_points)

        for idx, nodes in enumerate(hw_nodes):
            cluster_config[idx]["nodes"] = list(nodes)
            for node in nodes:
                node.meta["mase"].parameters["hardware"]["device_id"] = idx

    return mg


def partition_to_multi_device_transform_pass(
    mg, pass_args={"cluster_config": None, "device_count": 1, "mode": "naive"}
):
    """partition to multi device

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass does not need any arguments, defaults to None
    :type pass_args: _type_, optional, "cluster_config" specifies which devices the model is mapped to, defaults to None; "device_count" specifies the number of devices in the system, defaults to 1;  "mode" controls which algorithm is used for device-level partitioning, default to "naive"
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)


    This pass maps a model onto a hardware system consisting of multiple devices.
    This is useful when a model is too large and cannot fit onto a single device.
    The current version contains the following algorithms:

    - naive:
        - this algorithm simply maps each mase node onto a single device if the total number of devices is no less than the total number of mase nodes
        - if the total number of nodes is larger than the total number of devices, it will randomly insert cut points
    """

    cluster_config = pass_args["cluster_config"]
    device_count = pass_args["device_count"]
    mode = pass_args["mode"]

    if cluster_config is None:
        cluster_config = default_cluster_config(device_count)

    if mode == "naive":
        mg = apply_naive_partition(
            mg, cluster_config=cluster_config, device_count=device_count
        )

    return mg, {}
