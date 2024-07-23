import logging
import os
import shutil

from chop.passes.graph.utils import vf, init_project
from .logicnets.emit_linear import LogicNetsLinearVerilog
from .util import include_ip_to_project

logger = logging.getLogger(__name__)

from pathlib import Path


def _append(list1, list2):
    return list1 + list(set(list2) - set(list1))


def emit_internal_rtl_transform_pass(graph, pass_args={}):
    """
    Emit internal components
    """

    logger.info("Emitting internal components...")
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )

    init_project(project_dir)
    rtl_dir = os.path.join(project_dir, "hardware", "rtl")

    rtl_dependencies = []

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        if "INTERNAL_RTL" == node.meta["mase"].parameters["hardware"]["toolchain"]:
            if (
                hasattr(node.meta["mase"].module, "config")
                and isinstance(node.meta["mase"].module.config, dict)
                and node.meta["mase"].module.config.get("name", "") == "logicnets"
            ):
                # LogicNets hardware is generated programmatically from a mase node
                logger.info("Emitting LogicNets components...")
                node_name = vf(node.name)
                emitter = LogicNetsLinearVerilog(node.meta["mase"].module)
                emitter.gen_layer_verilog(node_name, rtl_dir)
            else:
                # For other nodes, simply include the corresponding written IP
                files = include_ip_to_project(node)
                rtl_dependencies = _append(rtl_dependencies, files)
        elif "INTERNAL_HLS" in node.meta["mase"].parameters["hardware"]["toolchain"]:
            assert False, "Internal HLS not implemented yet."
        else:  # QOL change to log a warning. May be useful for adding future modules to mase hardware.
            logger.warning(f"Node {node.name} has no toolchain specified. Skipping...")

    import mase_components

    hardware_dir = mase_components.__path__[0]

    for f in rtl_dependencies:
        shutil.copy(os.path.join(hardware_dir, f), rtl_dir)

    return graph, {}
