from ..circuit import Circuit
import numpy as np


class AnsatzTests(object):
    # To run tests with this base class, the following variables need to be properly initialized in the child class:
    # self.ansatz

    def test_set_number_of_layers(self):
        # Given
        new_number_of_layers = 100

        # When
        self.ansatz.number_of_layers = new_number_of_layers

        # Then
        self.assertEqual(self.ansatz.number_of_layers, new_number_of_layers)

    def test_set_number_of_layers_invalidates_parametrized_circuit(self):
        # Given
        new_number_of_layers = 100
        if self.ansatz.supports_parametrized_circuits:
            initial_circuit = self.ansatz.parametrized_circuit

            # When
            self.ansatz.number_of_layers = new_number_of_layers

            # Then
            self.assertIsNone(self.ansatz._parametrized_circuit)

    def test_number_of_params_greater_than_0(self):
        # When
        number_of_params = self.ansatz.number_of_params

        # Then
        self.assertTrue(number_of_params > 0)

    def test_number_of_qubits_greater_than_0(self):
        # When
        n_qubits = self.ansatz.number_of_qubits

        # Then
        self.assertTrue(n_qubits > 0)

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

    def test_order_of_symbols_is_the_same_for_generated_circuitget_symbols(self):
        # Given
        circuit_symbols = self.ansatz.parametrized_circuit.symbolic_params
        ansatz_symbols = self.ansatz.get_symbols

        self.assertEqual(circuit_symbols, ansatz_symbols)