import logging
import os
import shutil

from chop.passes.graph.utils import vf, v2p, get_module_by_name, init_project
from .logicnets.emit_linear import LogicNetsLinearVerilog
from .util import include_ip_to_project

logger = logging.getLogger(__name__)


def emit_logicnets_transform_pass(graph, pass_args={}):
    """
    Emit LogicNets Components
    """

    logger.info("Emitting LogicNets components...")
    project_dir = (
        pass_args["project_dir"] if "project_dir" in pass_args.keys() else "top"
    )

    init_project(project_dir)
    rtl_dir = os.path.join(project_dir, "hardware", "rtl")

    rtl_dependencies = []

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        if "INTERNAL_RTL" == node.meta["mase"].parameters["hardware"]["toolchain"]:
            node_name = vf(node.name)
            emitter = LogicNetsLinearVerilog(node.meta["mase"].module)
            emitter.gen_layer_verilog(node_name, rtl_dir)
            # files = include_ip_to_project(node)
            # rtl_dependencies = _append(rtl_dependencies, files)
        elif "INTERNAL_HLS" in node.meta["mase"].parameters["hardware"]["toolchain"]:
            assert False, "Intenral HLS not implemented yet."

    hardware_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "components",
    )

    for f in rtl_dependencies:
        shutil.copy(os.path.join(hardware_dir, f), rtl_dir)

    return graph
