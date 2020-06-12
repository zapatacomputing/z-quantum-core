import unittest
import json
import networkx as nx
from .graph import (
    compare_graphs,
    save_graph,
    load_graph,
    generate_graph_node_dict,
    generate_random_graph_erdos_renyi,
)
import os


class TestGraph(unittest.TestCase):
    def test_compare_graphs(self):
        # Given
        G1 = nx.Graph()
        G2 = nx.Graph()
        G3 = nx.Graph()

        G1.add_edges_from([(1, 2), (2, 3), (1, 3)])
        G2.add_edges_from([(1, 2), (2, 3), (1, 3)])
        G3.add_edges_from([(1, 2), (2, 3)])

        # When/Then
        self.assertTrue(compare_graphs(G1, G2))
        self.assertFalse(compare_graphs(G2, G3))

    def test_graph_io(self):
        # Given
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (1, 3)])

        # When
        save_graph(G, "Graph.json")
        G2 = load_graph("Graph.json")

        # Then
        self.assertTrue(compare_graphs(G, G2))

        os.remove("Graph.json")

    def test_generate_graph_node_dict(self):
        # Given
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (1, 3)])
        target_node_dict = {1: 0, 2: 1, 3: 2}

        # When
        node_dict = generate_graph_node_dict(G)

        # Then
        self.assertDictEqual(node_dict, target_node_dict)

    def test_generate_random_graph_erdos_renyi(self):
        # Given
        num_nodes = 3
        probability = 1
        target_graph = nx.Graph()
        target_graph.add_edges_from([(0, 1), (1, 2), (0, 2)])

        # When
        graph = generate_random_graph_erdos_renyi(num_nodes, probability)

        # Then
        self.assertTrue(compare_graphs(graph, target_graph))

        # Given
        num_nodes = 3
        probability = 0
        target_graph = nx.Graph()

        # When
        graph = generate_random_graph_erdos_renyi(num_nodes, probability)

        # Then
        self.assertTrue(compare_graphs(graph, target_graph))

        # Given
        num_nodes = 20
        probability = 0.8
        random_weights = True

        # When
        graph = generate_random_graph_erdos_renyi(
            num_nodes, probability, random_weights
        )

        # Then
        for edge in graph.edges:
            self.assertIn("weight", graph.edges[edge].keys())

        self.assertEqual(len(graph.nodes), num_nodes)
