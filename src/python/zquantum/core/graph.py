import json
import random
import warnings
from typing import Callable, Dict, Optional

import networkx as nx

from .serialization import ensure_open
from .typing import DumpTarget, LoadSource
from .utils import SCHEMA_VERSION, create_object

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
    """
    nodes_int_map = []
    for node_index, node in enumerate(graph.nodes):
        nodes_int_map.append((node, node_index))
    nodes_dict = dict(nodes_int_map)
    return nodes_dict


Sampler = Callable[[], float]


def static_sampler() -> Sampler:
    def _sample_next():
        return 1.0

    return _sample_next


def uniform_sampler(min_value=0, max_value=1) -> Sampler:
    def _sample_next():
        return random.uniform(min_value, max_value)

    return _sample_next


def constant_sampler(value) -> Sampler:
    def _sample_next():
        return value

    return _sample_next


def generate_random_graph_erdos_renyi(
    num_nodes: int,
    connection_sampler: Sampler,
    weight_sampler: Sampler = static_sampler(),
    seed: Optional[int] = None,
) -> nx.Graph:
    """Randomly generate a graph from Erdos-Renyi ensemble.
    A graph is constructed by connecting nodes randomly.
    Each edge is included in the graph with probability p independent from
    every other edge. Equivalently, all graphs with n nodes and M edges have
    equal probability.

    Args:
        num_nodes: Number of nodes in the result graph.
        connection_sampler: Used to sample probability of two nodes connecting.
        weight_sampler: Used to sample edge weights. Defaults `static_sampler`,
            i.e. all edge weights are set to 1.0.
        seed: if provided, sets the global seed
    """
    probability = connection_sampler()
    output_graph = nx.erdos_renyi_graph(n=num_nodes, p=probability, seed=seed)
    _weight_graph_edges(output_graph, weight_sampler, seed)

    return output_graph


def generate_random_regular_graph(
    num_nodes: int,
    degree: int,
    weight_sampler: Sampler = static_sampler(),
    seed: Optional[int] = None,
) -> nx.Graph:
    """Randomly generate a d-regular graph.
    A graph is generated by picking uniformly a graph among the set of graphs
    with the desired number of nodes and degree.
    Args:
        num_nodes: Number of nodes in the generated graph.
        degree: Degree of each edge.
        weight_sampler: Used to sample edge weights. Defaults `static_sampler`,
            i.e. all edge weights are set to 1.0.
        seed: if provided, sets the global seed
    """
    output_graph = nx.random_regular_graph(d=degree, n=num_nodes, seed=seed)
    _weight_graph_edges(output_graph, weight_sampler, seed)

    return output_graph


def generate_caveman_graph(
    number_of_cliques: int,
    size_of_cliques: int,
    weight_sampler: Sampler = static_sampler(),
    seed: Optional[int] = None,
) -> nx.Graph:
    output_graph = nx.caveman_graph(number_of_cliques, size_of_cliques)
    _weight_graph_edges(output_graph, weight_sampler, seed)
    return output_graph


def generate_ladder_graph(
    length_of_ladder: int,
    weight_sampler: Sampler = static_sampler(),
    seed: Optional[int] = None,
) -> nx.Graph:
    graph = nx.ladder_graph(length_of_ladder)
    _weight_graph_edges(graph, weight_sampler, seed)
    return graph


def generate_barbell_graph(
    number_of_vertices_complete_graph: int,
    weight_sampler: Sampler = static_sampler(),
    seed: Optional[int] = None,
) -> nx.Graph:
    graph = nx.barbell_graph(number_of_vertices_complete_graph, 0)
    _weight_graph_edges(graph, weight_sampler, seed)

    return graph


def _weight_graph_edges(
    graph: nx.Graph,
    sampler: Sampler,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Update the weights of all the edges of a graph in place.

    Args:
        graph: The graph to mutate.
        sampler:
    """
    if graph.is_multigraph():
        raise ValueError("Cannot deal with multigraphs")

    if seed is not None:
        random.seed(seed)

    weighted_edges = [(e[0], e[1], sampler()) for e in graph.edges]

    # If edges already present, it will effectively update them (except for multigraph)
    graph.add_weighted_edges_from(weighted_edges)


def generate_graph_from_specs(graph_specs: dict) -> nx.Graph:
    """(Deprecated) select graph generator by its name and arguments in the graph specs.

    This is deprecated because each graph generator function requires slightly different
    arguments, so wrapping all of them is convoluted and results in deeply nested dicts.

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
                - "type_graph" (str)
                - "num_nodes" (int)
                - "random_weights" (bool) - if True, weights are sampled from U(0, 1).
                    Defaults to False.
                - additional arguments to graph generator depending on the choice of
                    "type_graph" (see above)
            Optional keys:
                - "seed"

    """
    warnings.warn(
        "zquantum.core.generate_graph_from_specs is deprecated. Please use other, "
        "specialized functions from this module directly.",
        DeprecationWarning,
    )

    type_graph = graph_specs["type_graph"]
    num_nodes = graph_specs["num_nodes"]
    if graph_specs.get("random_weights", False):
        weight_sampler = uniform_sampler(0, 1)
    else:
        weight_sampler = static_sampler()

    seed = graph_specs.get("seed")

    if type_graph == "erdos_renyi":
        proba_sampler = constant_sampler(graph_specs["probability"])
        graph = generate_random_graph_erdos_renyi(
            num_nodes,
            proba_sampler,
            weight_sampler,
            seed,
        )

    elif type_graph == "regular":
        degree = graph_specs["degree"]
        graph = generate_random_regular_graph(num_nodes, degree, weight_sampler, seed)

    elif type_graph == "complete":
        connection_sampler = constant_sampler(1.0)
        graph = generate_random_graph_erdos_renyi(
            num_nodes, connection_sampler, weight_sampler, seed
        )

    elif type_graph == "caveman":
        number_of_cliques = graph_specs["number_of_cliques"]
        size_of_cliques = graph_specs["size_of_cliques"]
        graph = generate_caveman_graph(
            number_of_cliques,
            size_of_cliques,
            weight_sampler,
            seed,
        )

    elif type_graph == "ladder":
        length_of_ladder = graph_specs["length_of_ladder"]
        graph = generate_ladder_graph(length_of_ladder, weight_sampler, seed)

    elif type_graph == "barbell":
        n_vertices = graph_specs["number_of_vertices_complete_graph"]
        graph = generate_barbell_graph(
            number_of_vertices_complete_graph=n_vertices,
            weight_sampler=weight_sampler,
            seed=seed,
        )
    else:
        raise (NotImplementedError("This type of graph is not supported: ", type_graph))

    return graph
