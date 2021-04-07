"""Test case prototypes that can be used in other projects.

Note that this file won't be executed on its own by pytest.
You need to define your own test cases that inherit from the ones defined here.
"""


import numpy as np


class AnsatzTests:
    def test_set_number_of_layers(self, ansatz):
        # Given
        new_number_of_layers = 100
        # When
        ansatz.number_of_layers = new_number_of_layers
        # Then
        assert ansatz.number_of_layers == new_number_of_layers

    def test_set_number_of_layers_invalidates_parametrized_circuit(self, ansatz):
        # Given
        new_number_of_layers = 100
        if ansatz.supports_parametrized_circuits:
            ansatz.parametrized_circuit

            # When
            ansatz.number_of_layers = new_number_of_layers

            # Then
            assert ansatz._parametrized_circuit is None

    # TODO: check with QCBM?
    def test_number_of_params_greater_than_0(self, ansatz):
        if ansatz.number_of_layers != 0:
            assert ansatz.number_of_params >= 0

    def test_number_of_qubits_greater_than_0(self, ansatz):
        assert ansatz.number_of_qubits > 0

    def test_get_executable_circuit_is_not_empty(self, ansatz):
        # Given
        params = np.random.random([ansatz.number_of_params])

        # When
        circuit = ansatz.get_executable_circuit(params)

        # Then
        assert len(circuit.gates) > 0

    def test_get_executable_circuit_does_not_contain_symbols(self, ansatz):
        # Given
        params = np.random.random([ansatz.number_of_params])

        # When
        circuit = ansatz.get_executable_circuit(params=params)

        # Then
        for gate in circuit.gates:
            assert len(gate.symbolic_params) == 0
