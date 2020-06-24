from ..circuit import Circuit


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

    def test_generate_circuit(self):
        # When
        circuit = self.ansatz._generate_circuit()

        # Then
        self.assertTrue(isinstance(circuit, Circuit))
