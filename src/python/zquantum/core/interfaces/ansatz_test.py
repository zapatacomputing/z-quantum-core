from ..circuit import Circuit
import numpy as np


class AnsatzTests(object):
    # To run tests with this base class, the following variables need to be properly initialized in the child class:
    # self.ansatz

    def test_set_n_layers(self):
        # Given
        new_n_layers = 100

        # When
        self.ansatz.n_layers = new_n_layers

        # Then
        self.assertEqual(self.ansatz.n_layers, new_n_layers)

    def test_set_n_layers_invalidates_parametrized_circuit(self):
        # Given
        new_n_layers = 100
        if self.ansatz.supports_parametrized_circuits:
            initial_circuit = self.ansatz.parametrized_circuit

            # When
            self.ansatz.n_layers = new_n_layers

            # Then
            self.assertIsNone(self.ansatz._parametrized_circuit)

    def test_number_of_params(self):
        # When
        n_params = self.ansatz.number_of_params

        # Then
        self.assertTrue(n_params > 0)

    def test_get_executable_circuit_is_not_empty(self):
        # Given
        params = np.random.random([self.ansatz.number_of_params])

        # When
        circuit = self.ansatz.get_executable_circuit(params)

        # Then
        self.assertTrue(len(circuit.gates) > 0)

    def test_get_executable_circuit_does_not_contain_symbols(self):
        # Given
        params = np.random.random([self.ansatz.number_of_params])

        # When
        circuit = self.ansatz.get_executable_circuit(params=params)

        # Then
        for gate in circuit.gates:
            self.assertEqual(len(gate.symbolic_params), 0)
