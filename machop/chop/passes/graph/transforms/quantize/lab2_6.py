def print_arg(node, arg_name):
        print(node.meta["mase"].parameters["common"]["args"][arg_name]["type"])
        print(node.meta["mase"].parameters["common"]["args"][arg_name]["precision"])


MASE_OP_TO_INPUT_ENTRIES_AND_ARGS = {
    # entry and arg corresponding to name in software and hardware mapping
    "add": (("data_in", "data_in"), ("data_in_0", "data_in_1")),
    "bmm": (("data_in", "weight"), ("data_in_0", "data_in_1")),
    "conv1d": (("data_in", "weight", "bias"), ("data_in_0", "weight", "bias")),
    "conv2d": (("data_in", "weight", "bias"), ("data_in_0", "weight", "bias")),
    "matmul": (("data_in", "weight"), ("data_in_0", "data_in_1")),
    "mul": (("data_in", "data_in"), ("data_in_0", "data_in_1")),
    "linear": (("data_in", "weight", "bias"), ("data_in_0", "weight", "bias")),
    "relu": (("data_in",), ("data_in_0",)),
    "sub": (("data_in", "data_in"), ("data_in_0", "data_in_1")),
}

def arg_exists(node, arg_name) -> bool:
    return arg_name in node.meta["mase"].parameters["common"]["args"]


def print_node_meta_param(node, mase_op: str) -> None:
    
    for entry, arg in zip(*MASE_OP_TO_INPUT_ENTRIES_AND_ARGS[mase_op]):
        if not arg_exists(node, arg):
            continue
        print_arg(
            node,
            arg_name=arg,
        )



def graph_iterator_quantize_verify(graph):
    for node in graph.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        node_args = node.meta["mase"].parameters["common"]["args"]

        for arg_name, arg_val in node_args : 
            print(arg_val["type"])
            print(arg_val["precision"])
    

