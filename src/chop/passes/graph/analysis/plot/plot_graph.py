import networkx as nx
import matplotlib.pyplot as plt
from torch.fx.passes import graph_drawer


def plot_graph_analysis_pass(
    graph,
    pass_args={
        "file_name": None,
    },
):
    graph.draw(pass_args["file_name"])
    # nx_graph = nx.DiGraph()

    # for node in graph.nodes:
    #     nx_graph.add_node(str(node), label=node.name + "\n" + node.op)
    #     for user in node.users:
    #         nx_graph.add_edge(str(node), str(user))

    # pos = nx.spring_layout(nx_graph)  # Layout for our nodes
    # labels = nx.get_node_attributes(nx_graph, 'label')
    # plt.figure(figsize=(12, 8))
    # nx.draw(nx_graph, with_labels=True)
    # # nx.draw(nx_graph, pos, with_labels=True, labels=labels, node_size=3000, node_color='#ffcccc', font_size=10, font_weight='bold')
    # plt.title("Torch.fx Graph")
    # plt.show()
    return graph, {}
