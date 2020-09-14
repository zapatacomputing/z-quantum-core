from zquantum.core.graph import (
    generate_random_graph_erdos_renyi,
    generate_random_regular_graph,
    generate_graph_from_specs,
    save_graph,
)
from json import loads


def generate_erdos_renyi_graph(
    number_of_nodes, edge_probability, random_weights=False, seed="None"
):
    if seed == "None":
        seed = None
    graph = generate_random_graph_erdos_renyi(
        number_of_nodes, edge_probability, random_weights, seed
    )
    save_graph(graph, "graph.json")


def generate_regular_graph(number_of_nodes, degree, random_weights=False, seed="None"):
    if seed == "None":
        seed = None
    graph = generate_random_regular_graph(number_of_nodes, degree, random_weights, seed)
    save_graph(graph, "graph.json")


def generate_complete_graph(number_of_nodes, random_weights=False, seed="None"):
    if seed == "None":
        seed = None
    graph = generate_random_graph_erdos_renyi(
        number_of_nodes, 1.0, random_weights, seed
    )
    save_graph(graph, "graph.json")


def generate_custom_graph(graph_specs):
    graph = generate_graph_from_specs(graph_specs)
    save_graph(graph, "graph.json")
