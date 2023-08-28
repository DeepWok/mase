import numpy as np


def total_bits_mg_analysis_pass(graph, pass_args: dict):
    """
    Profile statistics analysis pass
    """

    data_in_cost, weights_cost = 0, 0
    data_in_size, weights_size = 0, 0

    for node in graph.nodes:
        mase_meta = node.meta["mase"].parameters
        mase_op = mase_meta["common"]["mase_op"]
        mase_type = mase_meta["common"]["mase_type"]

        if mase_type in ["module", "module_related_func"]:
            if mase_op in ["linear", "conv2d", "conv1d"]:
                data_in_0_meta = mase_meta["common"]["args"]["data_in_0"]
                w_meta = mase_meta["common"]["args"]["weight"]
                # maybe add bias
                d_size = np.prod(data_in_0_meta["size"])
                w_size = np.prod(w_meta["size"])
                data_in_cost += sum(data_in_0_meta["precision"]) * d_size
                data_in_size += d_size
                weights_size += w_size
                weights_cost += sum(w_meta["precision"]) * w_size
    # on average how many bits do we pay per value?
    data_avg_bit = data_in_cost / data_in_size
    w_avg_bit = weights_cost / weights_size
    return {"graph": graph, "data_avg_bit": data_avg_bit, "w_avg_bit": w_avg_bit}
