import ast
import json
import random
from random import choice, normalvariate, uniform
from typing import Optional, Union

import networkx as nx

from .serialization import ensure_open
from .typing import DumpTarget, LoadSource
from .utils import SCHEMA_VERSION

GRAPH_SCHEMA = SCHEMA_VERSION + "-graph"


def save_graph(graph: nx.Graph, filename: DumpTarget):
    """Saves a NetworkX graph object to JSON file.

    Args:
        graph: the input graph object
        filename: name of the output file
    """
    with ensure_open(filename, "w") as f:
        graph_dict = {
            **nx.readwrite.json_graph.node_link_data(graph),
            "schema": GRAPH_SCHEMA,
        }
        json.dump(graph_dict, f, indent=2)


def load_graph(file: LoadSource) -> nx.Graph:
    """Reads a JSON file for extracting the NetworkX graph object.

    Args:
        file: the file or filepath to load
    """

    with ensure_open(file) as f:
        data = json.load(f)

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
    num_nodes: int,
    probability: float,
    weights: str = "static",
    seed: Optional[int] = None,
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
        weights: str
            String indicating how the edge weights should are assigned.
            By default "static", i.e. all the edge weights are set to 1.
            More details on how to specify random distributions in weight_graph_edges()


    Returns:
        A networkx.Graph object
    """
    output_graph = nx.erdos_renyi_graph(n=num_nodes, p=probability, seed=seed)
    _weight_graph_edges(output_graph, weights, seed)

    return output_graph


def generate_random_regular_graph(
    num_nodes: int,
    degree: int,
    weights: str = "static",
    seed: Optional[int] = None,
) -> nx.Graph:
    """Randomly generate a d-regular graph.
    A graph is generated by picking uniformly a graph among the set of graphs
    with the desired number of nodes and degree.
    Args:
        num_nodes: integer
            Number of nodes.
        degree: int
            Degree of each edge.
        weights: str
            String indicating how the edge weights should are assigned.
            By default "static", i.e. all the edge weights are set to 1.
            More details on how to specify random distributions in weight_graph_edges()

    Returns:
        A networkx.Graph object
    """
    output_graph = nx.random_regular_graph(d=degree, n=num_nodes, seed=seed)
    _weight_graph_edges(output_graph, weights, seed)

    return output_graph


def generate_caveman_graph(
    number_of_cliques: int,
    size_of_cliques: int,
    weights: str = "static",
    seed: Optional[int] = None,
) -> nx.Graph:
    output_graph = nx.caveman_graph(number_of_cliques, size_of_cliques)
    _weight_graph_edges(output_graph, weights, seed)
    return output_graph


def generate_ladder_graph(
    length_of_ladder: int, weights: str = "static", seed: Optional[int] = None
) -> nx.Graph:
    output_graph = nx.ladder_graph(length_of_ladder)
    _weight_graph_edges(output_graph, weights, seed)
    return output_graph


def generate_barbell_graph(
    number_of_vertices_complete_graph: int,
    weights: str = "static",
    seed: Optional[int] = None,
) -> nx.Graph:
    output_graph = nx.barbell_graph(number_of_vertices_complete_graph, 0)
    _weight_graph_edges(output_graph, weights, seed)
    return output_graph


def _weight_graph_edges(
    graph: nx.Graph,
    weights: str = "static",
    seed: Optional[int] = None,
) -> nx.Graph:
    """Update the weights of all the edges of a graph.

    Args:
        graph: nx.Graph
            The input graph.
        weights: str
            String indicating how the edge weights should are assigned.
            By default "static", i.e. all the edge weights are set to 1.
            More details on how to specify random distributions in weight_graph_edges()
    """
    assert not (graph.is_multigraph()), "Cannot deal with multigraphs"
    if seed is not None:
        random.seed(seed)

    weighted_edges = [
        (e[0], e[1], _generate_random_value_from_string(weights)) for e in graph.edges
    ]

    # If edges already present, it will effectively update them (except for multigraph)
    graph.add_weighted_edges_from(weighted_edges)


def generate_graph_from_specs(graph_specs: dict) -> nx.Graph:
    """Generate a graph from a specs dictionary.

    Graph generation is controlled via the "type_graph" key. Each graph generator
    requires additional arguments passed as additional keys in the `graph_specs` dict.
    The available `type_graph` values are:
        - "erdos_renyi". Additional keys:
            - "probability" (int)
        - "regular". Additional keys:
            - "degree" (int)
        - "complete". Same as "erdos_renyi" with probability set to 1.0.
        - "caveman". Additional keys:
            - "number_of_cliques" (int)
            - "size_of_cliques" (int)
        - "ladder". Additional keys:
            - "length_of_ladder" (int)
        - "barbell". Additional keys:
            - "number_of_vertices_complete_graph" (int)

    Args:
        graph_specs: specification of graphs to generate. Required keys:
                - "type_graph"
                - "num_nodes"
                - additional arguments to graph generator depending on the choice of
                    "type_graph" (see above)
            Optional keys:
                - "weights", defaults to "static"
                - "seed'
    """
    type_graph = graph_specs["type_graph"]
    num_nodes = graph_specs["num_nodes"]
    weights = graph_specs.get("weights", "static")
    seed = graph_specs.get("seed")

    if type_graph == "erdos_renyi":
        probability = graph_specs["probability"]
        if isinstance(probability, str):
            probability = _generate_random_value_from_string(probability)
        graph = generate_random_graph_erdos_renyi(num_nodes, probability, weights, seed)

    elif type_graph == "regular":
        degree = graph_specs["degree"]
        if isinstance(degree, str):
            degree = _generate_random_value_from_string(degree)
        graph = generate_random_regular_graph(num_nodes, degree, weights, seed)

    elif type_graph == "complete":
        graph = generate_random_graph_erdos_renyi(num_nodes, 1.0, weights, seed)

    elif type_graph == "caveman":
        number_of_cliques = graph_specs["number_of_cliques"]
        size_of_cliques = graph_specs["size_of_cliques"]
        graph = generate_caveman_graph(
            number_of_cliques, size_of_cliques, weights, seed
        )

    elif type_graph == "ladder":
        length_of_ladder = graph_specs["length_of_ladder"]
        graph = generate_ladder_graph(length_of_ladder, weights, seed)

    elif type_graph == "barbell":
        number_of_vertices_complete_graph = graph_specs[
            "number_of_vertices_complete_graph"
        ]
        graph = generate_barbell_graph(number_of_vertices_complete_graph, weights, seed)
    else:
        raise (NotImplementedError("This type of graph is not supported: ", type_graph))

    return graph
