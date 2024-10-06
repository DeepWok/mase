

import chop as chop


from chop.tools.logger import set_logging_verbosity
from chop.passes.graph.transforms.quantize import QUANTIZEABLE_OP

set_logging_verbosity("debug")


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()
def update_common_metadata_pass(mg, quan_args):
    # There is a bug in the current quantization pass, where the results metadata is not updated with the precision.
    # # Here we update the metadata here so we can test the hardware back end.

    # update precision
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
    # update parameters
    for node in mg.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        if mase_op in ["layer_norm"]:
            if node.meta["mase"].parameters["common"]["args"].get("weight") != None:
                node.meta["mase"].parameters["common"]["args"][
                    "elementwise_affine"
                ] = True
                if node.meta["mase"].parameters["common"]["args"].get("bias") != None:
                    node.meta["mase"].parameters["common"]["args"]["has_bias"] = True
    
    # update for transpose
    for node in mg.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        if mase_op in ["vit_self_attention_integer", "linear"]:
            args = node.meta["mase"].parameters["common"]["args"] 
            if args != None:
                for key, _ in args.items():
                    if "weight" in key:
                        args[key]["value"] = args[key]["value"].transpose(0,1)
                        args[key]["shape"].reverse()
                
    


def manually_update_hardware_parallelism_param(graph, pass_args: dict = {}):
    # The quantization pass currently don't support any inlayer precision automatically generate
    # we only have data_in, weight.. param in common metadata
    # in order to support in layer fine grained precision tuning
    # we just update the hardware metadata directly.
    for node in list(graph.fx_graph.nodes) + graph.nodes_in + graph.nodes_out:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        vp = node.meta["mase"]["hardware"].get("verilog_param")
        if vp == None:
            continue
        for key, value in pass_args.items():
           if key in node.name:
                if mase_op == "linear":
                    # weight1 = in0
                    vp["DATA_IN_0_PARALLELISM_DIM_0"] = value["din"][1]
                    vp["DATA_IN_0_PARALLELISM_DIM_1"] = value["din"][0]
                    vp["WEIGHT_PARALLELISM_DIM_0"] = value["din"][1]
                    vp["WEIGHT_PARALLELISM_DIM_1"] = value["dout"][1]
                    vp["BIAS_PARALLELISM_DIM_0"] = value["dout"][1]
                    vp["BIAS_PARALLELISM_DIM_1"] = 1
                    vp["DATA_OUT_0_PARALLELISM_DIM_0"] = value["dout"][1]
                    vp["DATA_OUT_0_PARALLELISM_DIM_1"] = value["dout"][0]
                elif mase_op == "fork2":
                    vp["DATA_IN_0_PARALLELISM_DIM_0"] = value["din"][1]
                    vp["DATA_IN_0_PARALLELISM_DIM_1"] = value["din"][0]
                    vp["DATA_OUT_0_PARALLELISM_DIM_0"] = value["dout"][0][1]
                    vp["DATA_OUT_0_PARALLELISM_DIM_1"] = value["dout"][0][0]
                    vp["DATA_OUT_1_PARALLELISM_DIM_0"] = value["dout"][1][1]
                    vp["DATA_OUT_1_PARALLELISM_DIM_1"] = value["dout"][1][0]
                elif mase_op == "add":
                    vp["DATA_IN_0_PARALLELISM_DIM_0"] = value["din"][0][1]
                    vp["DATA_IN_0_PARALLELISM_DIM_1"] = value["din"][0][0]
                    vp["DATA_IN_1_PARALLELISM_DIM_0"] = value["din"][1][1]
                    vp["DATA_IN_1_PARALLELISM_DIM_1"] = value["din"][1][0]
                    vp["DATA_OUT_0_PARALLELISM_DIM_0"] = value["dout"][1]
                    vp["DATA_OUT_0_PARALLELISM_DIM_1"] = value["dout"][0]
                elif mase_op == "vit_self_attention_integer":
                    num_heads = vp["NUM_HEADS"] 
                    vp["DATA_IN_0_PARALLELISM_DIM_0"] = value["din"][1]
                    vp["DATA_IN_0_PARALLELISM_DIM_1"] = value["din"][0]
                    vp["QUERY_WEIGHT_PARALLELISM_DIM_0"] = value["dattn"][1]//num_heads
                    vp["QUERY_WEIGHT_PARALLELISM_DIM_1"] = value["din"][1]
                    vp["QUERY_BIAS_PARALLELISM_DIM_0"] = value["dattn"][1]//num_heads
                    vp["QUERY_BIAS_PARALLELISM_DIM_1"] = 1
                    vp["KEY_WEIGHT_PARALLELISM_DIM_0"] = value["dattn"][1]//num_heads
                    vp["KEY_WEIGHT_PARALLELISM_DIM_1"] = value["din"][1]
                    vp["KEY_BIAS_PARALLELISM_DIM_0"] = value["dattn"][1]//num_heads
                    vp["KEY_BIAS_PARALLELISM_DIM_1"] = 1
                    vp["VALUE_WEIGHT_PARALLELISM_DIM_0"] = value["dattn"][1]//num_heads
                    vp["VALUE_WEIGHT_PARALLELISM_DIM_1"] = value["din"][1]
                    vp["VALUE_BIAS_PARALLELISM_DIM_0"] = value["dattn"][1]//num_heads
                    vp["VALUE_BIAS_PARALLELISM_DIM_1"] = 1
                    vp["PROJ_WEIGHT_PARALLELISM_DIM_0"] = value["dout"][1] 
                    vp["PROJ_WEIGHT_PARALLELISM_DIM_1"] = value["dattn"][1]//num_heads
                    vp["PROJ_BIAS_PARALLELISM_DIM_0"] = value["dout"][1]
                    vp["PROJ_BIAS_PARALLELISM_DIM_1"] = 1
                    vp["DATA_OUT_0_PARALLELISM_DIM_0"] = value["dout"][1]
                    vp["DATA_OUT_0_PARALLELISM_DIM_1"] = value["dout"][0]
                elif mase_op == "layer_norm":
                    vp["DATA_IN_0_PARALLELISM_DIM_0"] = value["din"][1]
                    vp["DATA_IN_0_PARALLELISM_DIM_1"] = value["din"][0]
                    vp["WEIGHT_PARALLELISM_DIM_0"] = value["dout"][1]
                    vp["WEIGHT_PARALLELISM_DIM_1"] = 1
                    vp["BIAS_PARALLELISM_DIM_0"] = value["dout"][1]
                    vp["BIAS_PARALLELISM_DIM_1"] = 1
                    vp["DATA_OUT_0_PARALLELISM_DIM_0"] = value["dout"][1]
                    vp["DATA_OUT_0_PARALLELISM_DIM_1"] = value["dout"][0]
                else:
                    vp["DATA_IN_0_PARALLELISM_DIM_0"] = value["din"][1]
                    vp["DATA_IN_0_PARALLELISM_DIM_1"] = value["din"][0]
                    vp["DATA_OUT_0_PARALLELISM_DIM_0"] = value["dout"][1]
                    vp["DATA_OUT_0_PARALLELISM_DIM_1"] = value["dout"][0]

    return graph, {}
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
    for node in list(mg.fx_graph.nodes):
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        vp = node.meta["mase"]["hardware"].get("verilog_param")
        if vp == None:
            continue
        delete_dim_of_batch_size(vp)
        if mase_op not in (QUANTIZEABLE_OP + ("vit_self_attention_integer",)):
            continue
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

        

def delete_dim_of_batch_size(vp):
    pop_list = []
    for key, item in vp.items():
        if any(keyword in key for keyword in ["DATA_IN", "DATA_OUT"]):
            if key.endswith("2"):
                pop_list.append(key)
    [vp.pop(key) for key in pop_list]


