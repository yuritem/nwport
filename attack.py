import random
from tqdm import tqdm

from portrait_utils import network_portrait


def nodes_subset(graph, which="hub", n_nodes=None):
    """
        Args:
            graph: nx.Graph
            which: str
                see attack() function for details
            n_nodes: int | None
                number of nodes
    """
    if n_nodes is None:
        n_nodes = min(10, graph.number_of_nodes())
    if which == "hub":
        sorted_nodes = list(dict(sorted(graph.degree, key=lambda x: x[1], reverse=True)).keys())  # sorted according to node degree in descending order
        return sorted_nodes[:n_nodes]
    elif which == "lone":
        sorted_nodes = list(dict(sorted(graph.degree, key=lambda x: x[1])).keys())  # sorted according to node degree in descending order
        return sorted_nodes[:n_nodes]
    elif which == "random":
        return random.sample(graph.nodes(), k=n_nodes)
    else:
        raise NotImplementedError("'which' options other than 'hub', 'lone' and 'random' are not implemented.")


def attack(graph_, attack_type='hub', n_iter=None, graph_label=None, delete_nodes=False):
    """
        Funciton that attacks a network (deletes nodes one by one) in a way specified by the 'attack_type' arg.

        Args:
            graph_: nx.Graph
                Networkx graph object to attack.
            attack_type: str
                Type of the attack. Possible values:
                    - 'hub': attack highest degree nodes first
                    - 'lone': attack lowest degree nodes first
                    - 'random': attack random nodes
            n_iter: int | None
                Number of iterations. Set to 1/10 of graph's number of nodes by default.
            graph_label: str
                Name of the graph to label it on the figures.
            delete_nodes: bool
                Whether nodes get deleted or not.
                If False, all edges that are linked to the node are deleleted and the node is kept.
    """

    if n_iter is None:
        n_iter = graph_.number_of_nodes() // 10

    graph = graph_.copy()
    port_init = network_portrait(graph)

    if graph_label is None:
        graph_label = f"{attack_type}_attack_graph"

    graphs, portraits, labels = [graph, ], [port_init, ], [graph_label, ]

    pbar = tqdm(range(1, n_iter + 1), leave=True)
    for i in pbar:
        pbar.set_description("Attack")
        graph_tmp = graph.copy()
        nodes_to_remove = nodes_subset(graph_tmp, which=attack_type, n_nodes=i)
        if delete_nodes:
            graph_tmp.remove_nodes_from(nodes_to_remove)
        else:
            for node in nodes_to_remove:
                graph_tmp.remove_edges_from(graph.edges(node))
        graph_tmp_ = graph_tmp.copy()
        graphs.append(graph_tmp_)
        portraits.append(network_portrait(graph_tmp_))
        labels.append(f"{graph_label}_removed{i:03}")

    return graphs, portraits, labels
