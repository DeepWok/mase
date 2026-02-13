from chop.passes.graph.analysis.add_metadata.add_hardware_metadata import *
from collections import OrderedDict


def difflogic_hardware_metadata_optimize_pass(graph, args={}):

    def _is_logiclayer(node):
        return node.meta["mase"]["common"]["mase_op"] == "user_defined_module"

    for node in graph.nodes:
        if _is_logiclayer(node):
            pre_common_args_md = node.meta["mase"]["common"]["args"]
            post_common_args_md = {}
            node.meta["mase"]["hardware"]["difflogic_args"] = {}
            for k, v in pre_common_args_md.items():
                if "data_in" not in k:
                    node.meta["mase"]["hardware"]["difflogic_args"][k] = v
                else:
                    post_common_args_md[k] = v
            post_common_args_md = OrderedDict(post_common_args_md)
            node.meta["mase"]["common"]["args"] = post_common_args_md
    return (graph, None)


def difflogic_hardware_force_fixed_flatten_pass(graph, args={}):
    for node in graph.nodes:
        if node.meta["mase"]["common"]["mase_op"] == "flatten":
            # add_component source
            node.meta["mase"]["hardware"]["toolchain"] = "INTERNAL_RTL"
            node.meta["mase"]["hardware"]["module"] = "fixed_difflogic_flatten"
            node.meta["mase"]["hardware"]["dependence_files"] = [
                "difflogic_layers/rtl/fixed_difflogic_flatten.sv"
            ]
            # else
            add_verilog_param(node)
            add_extra_verilog_param(node, graph)
            graph.meta["mase"]["hardware"]["verilog_sources"] += node.meta["mase"][
                "hardware"
            ]["dependence_files"]
    return (graph, None)
