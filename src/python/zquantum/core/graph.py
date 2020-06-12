import networkx as nx
import json
from itertools import combinations
from random import uniform
import networkx as nx
from .utils import SCHEMA_VERSION
from typing import TextIO


def save_graph(graph: nx.Graph, filename: str):
    """Saves a NetworkX graph object to JSON file.

    Args:
        graph (networks.Graph): the input graph object
        filename (string): name of the output file
    """
    f = open(filename, "w")
    graph_dict = nx.readwrite.json_graph.node_link_data(graph)
    graph_dict["schema"] = SCHEMA_VERSION + "-graph"
    json.dump(graph_dict, f, indent=2)
    f.close()


def load_graph(file: TextIO) -> nx.Graph:
    """Reads a JSON file for extracting the NetworkX graph object.

    Args:
        file (str or file-like object): the file to load
    
    Returns:
        networkx.Graph: the graph
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return nx.readwrite.json_graph.node_link_graph(data)


def compare_graphs(graph1: nx.Graph, graph2: nx.Graph) -> bool:
    """Compares two NetworkX graph objects to see if they are identical.
    NOTE: this is *not* solving isomorphism problem.
    """

    for n1, n2 in zip(graph1.nodes, graph2.nodes):
        if n1 != n2:
            return False
    for e1, e2 in zip(graph1.edges, graph2.edges):
        if e1 != e2:
            return False
    return True


def generate_graph_node_dict(graph: nx.Graph) -> dict:
    """Generates a dictionary containing key:value pairs in the form of
                    nx.Graph node : integer index of the node
    
    Args:
        graph: nx.Graph object
    
    Returns:
        A dictionary as described
    """
    nodes_int_map = []
    for node_index, node in enumerate(graph.nodes):
        nodes_int_map.append((node, node_index))
    nodes_dict = dict(nodes_int_map)
    return nodes_dict


def generate_random_graph_erdos_renyi(
    num_nodes: int, probability: float, random_weights: bool = False
) -> nx.Graph:
    """Randomly generate a graph from Erdos-Renyi ensemble. 
    A graph is constructed by connecting nodes randomly. 
    Each edge is included in the graph with probability p independent from 
    every other edge. Equivalently, all graphs with n nodes and M edges have 
    equal probability.

    Args:
        num_nodes: integer
            Number of nodes.
        probability: float
            Probability of two nodes connecting.
        random_weights: bool
            Flag indicating whether the weights should be random or constant.
    
    Returns:
        A networkx.Graph object
    """

    output_graph = nx.Graph()
    output_graph.add_nodes_from(range(0, num_nodes))

    # iterate through all pairs of nodes
    for pair in combinations(range(0, num_nodes), 2):
        if uniform(0, 1) < probability:  # with the given probability
            if random_weights:
                weight = uniform(0, 1)
            else:
                weight = 1.0
            output_graph.add_edge(pair[0], pair[1], weight=weight)

    return output_graph
