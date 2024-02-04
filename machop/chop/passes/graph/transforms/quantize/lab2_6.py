import logging
import os
import pandas as pd
from tabulate import tabulate

logger = logging.getLogger(__name__)

def graph_iterator_quantize_verify(graph):
    headers = [
        "Graph name",
        "MASE_TYPE",
        "Mase_OP",
        "Argument Name",
        "Argument Type",
        "Argument Precision"
    ]
    rows = []
    for node in graph.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        node_args = node.meta["mase"].parameters["common"]["args"]
       
        if node.op == "call_module" or node.meta["mase"].parameters["common"]["mase_type"] in [
            "builtin_func",
            "module_related_func",
        ]:
       
          for arg_name, arg_val in node_args.items() : 
              rows.append(
                [
                node.name,
                node.meta["mase"].parameters["common"]["mase_type"],
                node.meta["mase"].parameters["common"]["mase_op"],
                arg_name,
                arg_val["type"],
                arg_val["precision"],
                ])  
        
    df = pd.DataFrame(rows, columns=headers)    
    logger.info("Compare nodes:")
    logger.info("\n" + tabulate(df, headers=headers, tablefmt="orgtbl"))

