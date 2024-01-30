import logging

logger = logging.getLogger(__name__)


def report_graph_analysis_pass(graph, pass_args={"file_name": None}):
    """
    <<<<<<< HEAD
        Generates a report for the graph analysis
    =======
        Generates a report for the graph analysis
    >>>>>>> main
        and prints out an over the model in a table.


        :param graph: a MaseGraph
    <<<<<<< HEAD
        :type graph: MaseGraph
        :param pass_args: this pass can take a string argument named "file_name", defaults to None
        :type pass_args: dict, optional
        :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
        :rtype: tuple(MaseGraph, dict)
    =======
        :type graph: MaseGraph
        :param pass_args: this pass can take a string argument named "file_name", defaults to None
        :type pass_args: dict, optional
        :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
        :rtype: tuple(MaseGraph, dict)
    >>>>>>> main

    """
    file_name = pass_args.get("file_name")
    buff = ""
    buff += str(graph.fx_graph)
    count = {
        "placeholder": 0,
        "get_attr": 0,
        "call_function": 0,
        "call_method": 0,
        "call_module": 0,
        "output": 0,
    }
    layer_types = []

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].module is not None:
            layer_types.append(node.meta["mase"].module)

    for node in graph.fx_graph.nodes:
        count[node.op] += 1
    buff += f"""Network overview:
{count}
Layer types:
{layer_types}"""
    if file_name is None:
        print(buff)
    else:
        with open(file_name, "w", encoding="utf-8") as outf:
            outf.write(buff)
    return graph, {}
