

import chop as chop


from chop.tools.logger import set_logging_verbosity
from chop.passes.graph.transforms.quantize import QUANTIZEABLE_OP

set_logging_verbosity("debug")

def update_common_metadata_pass(mg, quan_args):
    # There is a bug in the current quantization pass, where the results metadata is not updated with the precision.
    # # Here we update the metadata here so we can test the hardware back end.
    for node in mg.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        if mase_op not in QUANTIZEABLE_OP:
            print(mase_op)
            continue
        node_quan_config = quan_args.get(mase_op)["config"]
        for arg, _ in node.meta["mase"].parameters["common"]["args"].items():
            if (
                type(node.meta["mase"].parameters["common"]["args"][arg]) == dict
                and "type" in node.meta["mase"].parameters["common"]["args"][arg].keys()
            ):
                node.meta["mase"].parameters["common"]["args"][arg]["type"] = "fixed"
        for result, _ in node.meta["mase"].parameters["common"]["results"].items():
            if (
                type(node.meta["mase"].parameters["common"]["results"][result]) == dict
                and "type"
                in node.meta["mase"].parameters["common"]["results"][result].keys()
            ):
                node.meta["mase"].parameters["common"]["results"][result][
                    "type"
                ] = "fixed"
                node.meta["mase"].parameters["common"]["results"][result][
                    "precision"
                ] = [
                    node_quan_config["data_out_width"],
                    node_quan_config["data_out_frac_width"],
                ]

    for node in mg.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        if mase_op in ["layer_norm"]:
            if node.meta["mase"].parameters["common"]["args"].get("weight") != None:
                node.meta["mase"].parameters["common"]["args"][
                    "elementwise_affine"
                ] = True
                if node.meta["mase"].parameters["common"]["args"].get("bias") != None:
                    node.meta["mase"].parameters["common"]["args"]["has_bias"] = True


def update_hardware_precision_param(mg, quan_args, model_args: dict = {}):
    # The quantization pass currently don't support any inlayer precision automatically generate
    # we only have data_in, weight.. param in common metadata
    # in order to support in layer fine grained precision tuning
    # we just update the hardware metadata directly.
    def _cap(name):
        """
        capitalize a string
        """
        return str(name).upper()

    for node in mg.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        if mase_op not in (QUANTIZEABLE_OP + ("vit_self_attention_integer",)):
            continue
        vp = node.meta["mase"]["hardware"]["verilog_param"]
        node_quan_args = quan_args.get(mase_op)["config"]
        node_model_args = model_args.get(mase_op)
        if mase_op in ["vit_self_attention_integer", "layer_norm"]:
            for arg_name, arg_info in node_quan_args.items():
                _list = ["data_in", "data_out", "weight", "bias"]
                if any(
                    keyword in arg_name
                    for keyword in ["data_in", "data_out", "weight", "bias"]
                ):
                    continue
                if "width" not in arg_name:
                    continue
                cofig_str = arg_name.replace("frac_width", "precision_1")
                cofig_str = cofig_str.replace("width", "precision_0")
                vp[_cap(cofig_str)] = arg_info
            if node_model_args == None:
                continue
            for arg_name, arg_info in node_model_args.items():
                if type(arg_info) == bool:
                    vp[_cap(arg_name)] = 1 if arg_info else 0
                else:
                    vp[_cap(arg_name)] = arg_info

