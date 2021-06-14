import json
import os

import pytest

from steps import graph


@pytest.mark.parametrize(
    "step_fn,step_kwargs",
    [
        (
            graph.generate_random_graph_erdos_renyi,
            {"number_of_nodes": 12, "edge_probability": 0.2},
        ),
        (
            graph.generate_random_regular_graph,
            {"number_of_nodes": 10, "degree": 3},
        ),
        (
            graph.generate_complete_graph,
            {"number_of_nodes": 8},
        ),
        (
            graph.generate_caveman_graph,
            {"number_of_cliques": 20, "size_of_cliques": 10},
        ),
        (
            graph.generate_ladder_graph,
            {"length_of_ladder": 10},
        ),
        (
            graph.generate_barbell_graph,
            {"number_of_vertices_complete_graph": 14},
        ),
    ],
)
def test_generates_non_empty_file(step_fn, step_kwargs):
    step_fn(**step_kwargs)

    graph_path = "graph.json"
    assert os.path.exists(graph_path)

    with open(graph_path) as f:
        written_graph = json.load(f)

    assert len(written_graph) > 0

    os.remove(graph_path)
