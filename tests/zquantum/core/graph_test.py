import numbers
import os
import random

import networkx as nx
import numpy as np
import numpy.testing
import pytest
from zquantum.core.graph import (
    choice_sampler,
    compare_graphs,
    constant_sampler,
    generate_barbell_graph,
    generate_caveman_graph,
    generate_graph_from_specs,
    generate_graph_node_dict,
    generate_ladder_graph,
    generate_random_graph_erdos_renyi,
    generate_random_regular_graph,
    load_graph,
    normal_sampler,
    save_graph,
    uniform_sampler,
)


class TestGraph:
    def test_compare_graphs(self):
        # Given
        G1 = nx.Graph()
        G2 = nx.Graph()
        G3 = nx.Graph()

        G1.add_edges_from([(1, 2), (2, 3), (1, 3)])
        G2.add_edges_from([(1, 2), (2, 3), (1, 3)])
        G3.add_edges_from([(1, 2), (2, 3)])

        # When/Then
        assert compare_graphs(G1, G2)
        assert not compare_graphs(G2, G3)

    def test_graph_io(self):
        # Given
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (1, 3)])

        # When
        save_graph(G, "Graph.json")
        G2 = load_graph("Graph.json")

        # Then
        assert compare_graphs(G, G2)

        os.remove("Graph.json")

    def test_generate_graph_node_dict(self):
        # Given
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (1, 3)])
        target_node_dict = {1: 0, 2: 1, 3: 2}

        # When
        node_dict = generate_graph_node_dict(G)

        # Then
        assert node_dict == target_node_dict

    def test_generate_random_graph_erdos_renyi(self):
        # Given
        num_nodes = 3
        target_graph = nx.Graph()
        target_graph.add_edges_from([(0, 1), (1, 2), (0, 2)])

        # When
        graph = generate_random_graph_erdos_renyi(num_nodes, 1)

        # Then
        assert compare_graphs(graph, target_graph)

        # Given
        num_nodes = 3
        target_graph = nx.Graph()

        # When
        graph = generate_random_graph_erdos_renyi(num_nodes, 1)

        # Then
        assert compare_graphs(graph, target_graph)

        # Given
        num_nodes = 20
        weight_sampler = uniform_sampler()

        # When
        graph = generate_random_graph_erdos_renyi(num_nodes, 0.8, weight_sampler)

    def test_generate_random_regular_graph(self):
        # Given
        num_nodes = 4
        degree = 2

        # When
        graph = generate_random_regular_graph(num_nodes, degree)

        # Then
        for n in graph.nodes():
            node_in_edge = [n in e for e in graph.edges()]
            assert sum(node_in_edge) == degree

        # Given
        num_nodes = 20
        degree = 3
        weight_sampler = uniform_sampler()

        # When
        graph = generate_random_regular_graph(num_nodes, degree, weight_sampler)

        # Then
        for edge in graph.edges:
            assert "weight" in graph.edges[edge].keys()

        assert len(graph.nodes) == num_nodes

    def test_generate_caveman_graph(self):
        # Given
        number_of_cliques = 3
        size_of_cliques = 4
        weight_sampler = uniform_sampler()

        # When
        graph = generate_caveman_graph(
            number_of_cliques, size_of_cliques, weight_sampler
        )

        # Then
        for edge in graph.edges:
            assert "weight" in graph.edges[edge].keys()

        assert len(graph.nodes) == number_of_cliques * 4

    def test_generate_ladder_graph(self):
        # Given
        length_of_ladder = 4
        weight_sampler = uniform_sampler()

        # When
        graph = generate_ladder_graph(length_of_ladder, weight_sampler)

        # Then
        for edge in graph.edges:
            assert "weight" in graph.edges[edge].keys()

        assert len(graph.nodes) == length_of_ladder * 2

    def test_generate_barbell_graph(self):
        n_vertices = 4
        graph = generate_barbell_graph(n_vertices, uniform_sampler())

        for edge in graph.edges:
            assert "weight" in graph.edges[edge].keys()

        assert len(graph.nodes) == n_vertices * 2

    def test_seed(self):
        # Given
        num_nodes = 4
        degree = 2
        seed = 123
        weight_sampler = uniform_sampler()

        target_graph = generate_random_regular_graph(
            num_nodes, degree, weight_sampler=weight_sampler, seed=seed
        )

        # When
        graph = generate_random_regular_graph(
            num_nodes, degree, weight_sampler=uniform_sampler(), seed=seed
        )

        # Then
        assert compare_graphs(graph, target_graph)

    def test_generate_graph_from_specs(self):
        # Given
        specs = {"type_graph": "erdos_renyi", "num_nodes": 3, "probability": 1.0}
        target_graph = nx.Graph()
        target_graph.add_edges_from([(0, 1), (1, 2), (0, 2)])

        # When
        graph = generate_graph_from_specs(specs)

        # Then
        assert compare_graphs(graph, target_graph)

        # Given
        specs = {"type_graph": "regular", "num_nodes": 4, "degree": 2}

        # When
        graph = generate_graph_from_specs(specs)

        # Then
        for n in graph.nodes():
            node_in_edge = [n in e for e in graph.edges()]
            assert sum(node_in_edge) == 2

        # Given
        specs = {"type_graph": "complete", "num_nodes": 4}
        target_graph = nx.Graph()
        target_graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

        # When
        graph = generate_graph_from_specs(specs)

        # Then
        assert compare_graphs(graph, target_graph)

        # When
        specs = {"type_graph": "complete", "num_nodes": 10, "random_weights": True}

        # When
        graph = generate_graph_from_specs(specs)

        # Then
        for edge in graph.edges:
            assert "weight" in graph.edges[edge].keys()


@pytest.mark.parametrize(
    "sampler",
    [
        uniform_sampler(),
        uniform_sampler(2, 3),
        constant_sampler(-1),
        normal_sampler(2, 1),
        choice_sampler([2, 1, 6]),
    ],
)
class TestSamplers:
    def test_generates_numbers(self, sampler):
        assert isinstance(next(sampler), numbers.Number)

    def test_can_be_reused(self, sampler):
        next(sampler)
        next(sampler)

    def test_generates_same_number_when_seeded(self, sampler):
        samples = []
        for _ in range(100):
            random.seed(0)
            samples.append(next(sampler))

        np.testing.assert_array_equal(samples, samples[0])
