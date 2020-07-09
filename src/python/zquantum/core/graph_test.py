import unittest
import json
import networkx as nx
from .graph import (
    compare_graphs,
    save_graph,
    load_graph,
    generate_graph_node_dict,
    generate_random_graph_erdos_renyi,
    generate_random_graph_regular,
    generate_complete_graph,
    generate_graph_from_specs,
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

    def test_generate_random_graph_regular(self):
        # Given
        num_nodes = 4
        degree = 2

        # When
        graph = generate_random_graph_regular(num_nodes, degree)

        # Then
        for n in graph.nodes():
            node_in_edge = [n in e for e in graph.edges()]
            self.assertTrue(sum(node_in_edge) == degree)

        # Given
        num_nodes = 20
        degree = 3
        random_weights = True

        # When
        graph = generate_random_graph_regular(
            num_nodes, degree, random_weights
        )

        # Then
        for edge in graph.edges:
            self.assertIn("weight", graph.edges[edge].keys())

        self.assertEqual(len(graph.nodes), num_nodes)

    def test_generate_complete_graph(self):
        # Given
        num_nodes = 4

        target_graph = nx.Graph()
        target_graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        # When
        graph = generate_complete_graph(num_nodes)

        # Then
        self.assertTrue(compare_graphs(graph, target_graph))

        # Given
        num_nodes = 10
        random_weights = True

        # When
        graph = generate_complete_graph(
            num_nodes, random_weights
        )

        # Then
        for edge in graph.edges:
            self.assertIn("weight", graph.edges[edge].keys())

        self.assertEqual(len(graph.nodes), num_nodes)

    def test_generate_graph_from_specs(self):
        # Given
        specs = {'type_graph':'erdos_renyi', 'num_nodes':3, 'probability':1.}
        target_graph = nx.Graph()
        target_graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
        
        # When
        graph = generate_graph_from_specs(specs)

        # Then
        self.assertTrue(compare_graphs(graph, target_graph))

        # Given
        specs = {'type_graph':'regular', 'num_nodes':4 , 'degree':2}
    
        # When
        graph = generate_graph_from_specs(specs)
        
        # Then
        for n in graph.nodes():
            node_in_edge = [n in e for e in graph.edges()]
            self.assertTrue(sum(node_in_edge) == 2)
        
        # Given
        specs = {'type_graph':'complete', 'num_nodes': 4}
        target_graph = nx.Graph()
        target_graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        
        # When
        graph = generate_graph_from_specs(specs)

        # Then
        self.assertTrue(compare_graphs(graph, target_graph))
        
        # When
        specs = {'type_graph':'complete', 'num_nodes': 10, 'random_weights': True}


        # When
        graph = generate_graph_from_specs(specs)

        # Then
        for edge in graph.edges:
            self.assertIn("weight", graph.edges[edge].keys())

