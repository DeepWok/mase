from chop.tools.utils import nested_dict_replacer, to_numpy_if_tensor


def metadata_value_type_cast_transform_pass(graph, pass_args=None):
    """
    Apply a transformation to the MaseMetaData in given graph. All entries with a name "value" would be performed with a function

    :param graph: The input graph to be transformed.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    :return: The transformed graph.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.


    - pass_args
        - fn -> function: a function that performs certain conversions
    """

    fn = pass_args.pop("fn")
    for node in graph.fx_graph.nodes:
        node.meta["mase"].parameters = nested_dict_replacer(
            node.meta["mase"].parameters, fn
        )
    return graph, {}
