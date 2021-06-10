import json
from typing import Optional

import zquantum.core.graph
from zquantum.core.graph import save_graph
from zquantum.core.typing import Specs
from zquantum.core.utils import create_object


def _make_sampler(specs: Optional[Specs] = None) -> zquantum.core.graph.Sampler:
    if specs is None:
        return zquantum.core.graph.constant_sampler(1)

    elif isinstance(specs, str):
        sampler_dict = json.loads(specs)
        return create_object(sampler_dict)()

    elif isinstance(specs, dict):
        return create_object(specs)()

    else:
        raise ValueError(f"Invalid specs {specs}")


def generate_random_graph_erdos_renyi(
    number_of_nodes: int,
    edge_probability: float,
    sampler_specs: Optional[Specs] = None,
    seed: Optional[int] = None,
):
    graph = zquantum.core.graph.generate_random_graph_erdos_renyi(
        number_of_nodes,
        edge_probability,
        _make_sampler(sampler_specs),
        seed,
    )
    save_graph(graph, "graph.json")


def generate_random_regular_graph(
    number_of_nodes: int,
    degree: int,
    sampler_specs: Optional[Specs] = None,
    seed: Optional[int] = None,
):
    graph = zquantum.core.graph.generate_random_regular_graph(
        number_of_nodes,
        degree,
        _make_sampler(sampler_specs),
        seed,
    )
    save_graph(graph, "graph.json")


def generate_complete_graph(
    number_of_nodes: int,
    sampler_specs: Optional[Specs] = None,
    seed: Optional[int] = None,
):
    graph = zquantum.core.graph.generate_random_graph_erdos_renyi(
        number_of_nodes,
        1.0,
        _make_sampler(sampler_specs),
        seed,
    )
    save_graph(graph, "graph.json")


def generate_caveman_graph(
    number_of_cliques: int,
    size_of_cliques: int,
    sampler_specs: Optional[Specs] = None,
    seed: Optional[int] = None,
):
    graph = zquantum.core.graph.generate_caveman_graph(
        number_of_cliques,
        size_of_cliques,
        _make_sampler(sampler_specs),
        seed,
    )
    save_graph(graph, "graph.json")


def generate_ladder_graph(
    length_of_ladder: int,
    sampler_specs: Optional[Specs] = None,
    seed: Optional[int] = None,
):
    graph = zquantum.core.graph.generate_ladder_graph(
        length_of_ladder,
        _make_sampler(sampler_specs),
        seed,
    )
    save_graph(graph, "graph.json")


def generate_barbell_graph(
    number_of_vertices_complete_graph: int,
    sampler_specs: Optional[Specs] = None,
    seed: Optional[int] = None,
):
    graph = zquantum.core.graph.generate_barbell_graph(
        number_of_vertices_complete_graph,
        _make_sampler(sampler_specs),
        seed,
    )
    save_graph(graph, "graph.json")


def generate_graph_from_specs(graph_specs: str):
    graph_specs_dict = json.loads(graph_specs)
    graph = zquantum.core.graph.generate_graph_from_specs(graph_specs_dict)
    save_graph(graph, "graph.json")
