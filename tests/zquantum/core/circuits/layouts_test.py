import os

from zquantum.core.circuits.layouts import (
    CircuitConnectivity,
    CircuitLayers,
    build_circuit_layers_and_connectivity,
    load_circuit_connectivity,
    load_circuit_layers,
    load_circuit_ordering,
    save_circuit_connectivity,
    save_circuit_layers,
    save_circuit_ordering,
)


class TestCircuitLayers:
    def test_dict_io(self):
        # Given
        layers = CircuitLayers([[(0, 1), (2, 3)], [(1, 2), (3, 0)]])

        # When
        layers_dict = layers.to_dict()
        recreated_layers = CircuitLayers.from_dict(layers_dict)

        # Then
        assert len(layers.layers) == len(recreated_layers.layers)
        for layer, recreated_layer in zip(layers.layers, recreated_layers.layers):
            assert layer == recreated_layer

    def test_circuit_layers_io(self):
        # Given
        layers = CircuitLayers([[(0, 1), (2, 3)], [(1, 2), (3, 0)]])

        # When
        save_circuit_layers(layers, "layers.json")
        loaded_layers = load_circuit_layers("layers.json")

        # Then
        assert len(layers.layers) == len(loaded_layers.layers)
        for layer, loaded_layer in zip(layers.layers, loaded_layers.layers):
            assert layer == loaded_layer
        os.remove("layers.json")

    def test_circuit_ordering_io(self):
        # Given
        ordering = [0, 1, 3, 2]

        # When
        save_circuit_ordering(ordering, "ordering.json")
        loaded_ordering = load_circuit_ordering("ordering.json")

        # Then
        assert loaded_ordering == ordering
        os.remove("ordering.json")

    def test_circuit_connectivity_io(self):
        # Given
        connectivity = CircuitConnectivity([(0, 1), (1, 2), (2, 3), (3, 0)])
        # When
        save_circuit_connectivity(connectivity, "connectivity.json")
        loaded_connectivity = load_circuit_connectivity("connectivity.json")
        # Then
        assert len(connectivity.connectivity) == len(loaded_connectivity.connectivity)
        for connection, loaded_connection in zip(
            connectivity.connectivity, loaded_connectivity.connectivity
        ):
            assert connection == loaded_connection
        os.remove("connectivity.json")

    def test_build_circuit_layers_and_connectivity(self):
        # Sycamore
        # Given
        test_layers = [
            [(3, 1), (4, 7)],
            [(3, 6), (4, 2), (5, 8)],
            [(3, 1), (4, 2)],
            [(3, 6), (4, 7), (5, 8)],
            [(3, 0), (4, 8), (5, 2)],
            [(3, 7), (4, 1)],
            [(3, 0), (4, 1), (5, 2)],
            [(3, 7), (4, 8)],
        ]
        test_connectivity = [
            (3, 0),
            (3, 1),
            (4, 1),
            (4, 2),
            (5, 2),
            (3, 6),
            (3, 7),
            (4, 7),
            (4, 8),
            (5, 8),
        ]
        # When
        connectivity, layers = build_circuit_layers_and_connectivity(3, 3, "sycamore")

        # Then
        for layer, test_layer in zip(layers.layers, test_layers):
            assert layer == test_layer

        for row, test_row in zip(connectivity.connectivity, test_connectivity):
            assert row == test_row

        # Nearest-Neighbors
        # Given
        test_layers = [[(0, 1), (2, 3)], [(1, 2)]]
        test_connectivity = [(0, 1), (2, 3), (1, 2)]

        # When
        connectivity, layers = build_circuit_layers_and_connectivity(
            4, "nearest-neighbor"
        )

        # Then
        for layer, test_layer in zip(layers.layers, test_layers):
            assert layer == test_layer

        for row, test_row in zip(connectivity.connectivity, test_connectivity):
            assert row == test_row
