from zquantum.core.graph import (
    generate_random_graph_erdos_renyi as _generate_random_graph_erdos_renyi,
    generate_random_regular_graph as _generate_random_regular_graph,
    generate_caveman_graph as _generate_caveman_graph,
    generate_ladder_graph as _generate_ladder_graph,
    generate_barbell_graph as _generate_barbell_graph,
    generate_graph_from_specs as _generate_graph_from_specs,
    save_graph,
)
import json
from typing import Union, Dict, Optional


def generate_random_graph_erdos_renyi(
        number_of_nodes: int,
        edge_probability: float,
        random_weights: bool = False,
        seed: Optional[int] = None,
):
    graph = _generate_random_graph_erdos_renyi(
        number_of_nodes, edge_probability, random_weights, seed
    )
    save_graph(graph, "graph.json")


def generate_random_regular_graph(
        number_of_nodes: int,
        degree: int,
        random_weights: bool = False,
        seed: Optional[int] = None,
):
    graph = _generate_random_regular_graph(
        number_of_nodes, degree, random_weights, seed
    )
    save_graph(graph, "graph.json")


def generate_complete_graph(
        number_of_nodes: int, random_weights: bool = False, seed: Optional[int] = None
):
    graph = _generate_random_graph_erdos_renyi(
        number_of_nodes, 1.0, random_weights, seed
    )
    save_graph(graph, "graph.json")


def generate_caveman_graph(
        number_of_cliques: int, size_of_cliques: int, random_weights: bool = False, seed: Optional[int] = None
):
    graph = _generate_caveman_graph(
        number_of_cliques, size_of_cliques, random_weights, seed
    )
    save_graph(graph, "graph.json")


def generate_ladder_graph(
        length_of_ladder: int, random_weights: bool = False, seed: Optional[int] = None
):
    graph = _generate_ladder_graph(
        length_of_ladder, random_weights, seed
    )
    save_graph(graph, "graph.json")


def generate_barbell_graph(
        number_of_vertices_complete_graph: int, random_weights: bool = False,
        seed: Optional[int] = None
):
    graph = _generate_barbell_graph(
        number_of_vertices_complete_graph, random_weights, seed
    )
    save_graph(graph, "graph.json")


def generate_graph_from_specs(graph_specs: Dict):
    graph_specs_dict = json.loads(graph_specs)
    graph = _generate_graph_from_specs(graph_specs_dict)
    save_graph(graph, "graph.json")
