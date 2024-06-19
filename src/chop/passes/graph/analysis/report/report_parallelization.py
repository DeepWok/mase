
from tabulate import tabulate

def report_parallelization_analysis_pass(mg, pass_args={}):
    fname = pass_args.get("file_name", "report_parallelization.txt")

    headers = ["Node", "Node op", "Mase op", "Args", "Kwargs", "Valid Input Shardings", "Input Sharding", "Output Sharding"]
    info = []
    for node in mg.fx_graph.nodes:
        sharding_config = node.meta['mase']['software']['autosharding']
        info.append([
            node.name,
            node.op,
            node.meta['mase']['common']['mase_op'],
            node.args,
            node.kwargs,
            sharding_config["valid_input_shardings"],
            sharding_config["input_sharding"],
            sharding_config["output_sharding"]
        
        ])
    
    with open(fname, "w") as f:
        f.write(f"{tabulate(info, headers)}\n")
        
    return mg, {}