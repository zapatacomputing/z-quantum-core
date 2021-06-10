import json
from typing import Optional

import zquantum.core.graph
from zquantum.core.graph import save_graph


def _make_sampler(random_weights: bool = False):
    if random_weights:
        return zquantum.core.graph.uniform_sampler()
    else:
        return zquantum.core.graph.static_sampler()


def generate_random_graph_erdos_renyi(
    number_of_nodes: int,
    edge_probability: float,
    random_weights: bool = False,
    seed: Optional[int] = None,
):
    sampler = _make_sampler(random_weights)
    graph = zquantum.core.graph.generate_random_graph_erdos_renyi(
        number_of_nodes, edge_probability, sampler, seed
    )
    save_graph(graph, "graph.json")


def generate_random_regular_graph(
    number_of_nodes: int,
    degree: int,
    random_weights: bool = False,
    seed: Optional[int] = None,
):
    sampler = _make_sampler(random_weights)
    graph = zquantum.core.graph.generate_random_regular_graph(
        number_of_nodes, degree, sampler, seed
    )
    save_graph(graph, "graph.json")


def generate_complete_graph(
    number_of_nodes: int, random_weights: bool = False, seed: Optional[int] = None
):
    sampler = _make_sampler(random_weights)
    graph = zquantum.core.graph.generate_random_graph_erdos_renyi(
        number_of_nodes, 1.0, sampler, seed
    )
    save_graph(graph, "graph.json")


def generate_caveman_graph(
    number_of_cliques: int,
    size_of_cliques: int,
    random_weights: bool = False,
    seed: Optional[int] = None,
):
    sampler = _make_sampler(random_weights)
    graph = zquantum.core.graph.generate_caveman_graph(
        number_of_cliques, size_of_cliques, sampler, seed
    )
    save_graph(graph, "graph.json")


def generate_ladder_graph(
    length_of_ladder: int, random_weights: bool = False, seed: Optional[int] = None
):
    sampler = _make_sampler(random_weights)
    graph = zquantum.core.graph.generate_ladder_graph(length_of_ladder, sampler, seed)
    save_graph(graph, "graph.json")


def generate_barbell_graph(
    number_of_vertices_complete_graph: int,
    random_weights: bool = False,
    seed: Optional[int] = None,
):
    sampler = _make_sampler(random_weights)
    graph = zquantum.core.graph.generate_barbell_graph(
        number_of_vertices_complete_graph, sampler, seed
    )
    save_graph(graph, "graph.json")


def generate_graph_from_specs(graph_specs: str):
    graph_specs_dict = json.loads(graph_specs)
    graph = zquantum.core.graph.generate_graph_from_specs(graph_specs_dict)
    save_graph(graph, "graph.json")
