def verify(self):
    # Verify each node itself
    for node in self.fx_graph.nodes:
        node.meta["mase"].verify()

    # Each node must have a unique name and a unique verilog name
    node_names = []
    node_vf_names = []
    for node in self.fx_graph.nodes:
        assert node.name not in node_names
        assert vf(node.name) not in node_vf_names
        node_names.append(node.name)
        node_vf_names.append(vf(node.name))

    # Inter-node verification
    # Each edge between nodes must have the same size
    nodes_in = self.nodes_in
    nodes_out = self.nodes_out
    while nodes_in != nodes_out:
        next_nodes_in = []
        for node in nodes_in:
            for next_node, x in node.users.items():
                # This might have a bug - for now assume there is only one result
                if next_node.meta["mase"].parameters["hardware"]["is_implicit"]:
                    if node not in next_nodes_in:
                        next_nodes_in.append(node)
                    continue
                next_nodes_in.append(next_node)
                arg_count = len(next_node.all_input_nodes)
                if arg_count == 1:
                    assert (
                        next_node.meta["mase"].parameters["common"]["args"]["data_in"][
                            "size"
                        ]
                        == node.meta["mase"].parameters["common"]["results"][
                            "data_out"
                        ]["size"]
                    ), "Common input and output sizes mismatch: {} = {} and {} = {}".format(
                        node.name,
                        node.meta["mase"].parameters["common"]["results"]["data_out"][
                            "size"
                        ],
                        next_node.name,
                        next_node.meta["mase"].parameters["common"]["args"]["data_in"][
                            "size"
                        ],
                    )

                    assert (
                        next_node.meta["mase"].parameters["hardware"][
                            "verilog_parameters"
                        ]["IN_SIZE"]
                        == node.meta["mase"].parameters["hardware"][
                            "verilog_parameters"
                        ]["OUT_SIZE"]
                    ), "Verilog input and output sizes mismatch: {} = {} and {} = {}".format(
                        node.name,
                        node.meta["mase"].parameters["hardware"]["verilog_parameters"][
                            "OUT_SIZE"
                        ],
                        next_node.name,
                        next_node.meta["mase"].parameters["hardware"][
                            "verilog_parameters"
                        ]["IN_SIZE"],
                    )
                else:
                    i = get_input_index(node, next_node)
                    assert (
                        next_node.meta["mase"].parameters["common"]["args"][
                            f"data_in_{i}"
                        ]["size"]
                        == node.meta["mase"].parameters["common"]["results"][
                            "data_out"
                        ]["size"]
                    )
                    assert (
                        next_node.meta["mase"].parameters["hardware"][
                            "verilog_parameters"
                        ][f"IN_{i}_SIZE"]
                        == node.meta["mase"].parameters["hardware"][
                            "verilog_parameters"
                        ]["OUT_SIZE"]
                    ), "Verilog input and output sizes mismatch: {} = {} and {} = {}".format(
                        node.name,
                        node.meta["mase"].parameters["hardware"]["verilog_parameters"][
                            "OUT_SIZE"
                        ],
                        next_node.name,
                        next_node.meta["mase"].parameters["hardware"][
                            "verilog_parameters"
                        ][f"IN_{i}_SIZE"],
                    )
        assert (
            nodes_in != next_nodes_in
        ), f"Parsing error: cannot find the next nodes: {nodes_in}."
        nodes_in = next_nodes_in
    return self
