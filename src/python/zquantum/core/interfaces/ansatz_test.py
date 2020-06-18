class AnsatzTests(object):
    # To run tests with this base class, the following variables need to be properly initialized in the child class:
    # self.ansatz

    def test_set_gradient_type(self):
        # Given
        gradient_type = self.ansatz.supported_gradient_methods[0]

        # When
        sefl.ansatz.gradient_type = gradient_type

        # Then
        self.assertEqual(self.ansatz.gradient_type, gradient_type)

    def test_set_gradient_type_throws_error(self):
        # Given
        incorrect_gradient_type = "test"

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz.gradient_type = incorrect_gradient_type

    def test_set_gradient_type_invalidates_circuits(self):
        # Given
        gradient_type = self.ansatz.supported_gradient_methods[0]

        # When
        self.ansatz.gradient_type = gradient_type

        # Then
        self.assertIsNone(self.ansatz._gradient_circuits)

    def test_set_n_qubits(self):
        # Given
        new_n_qubits = 100

        # When
        self.ansatz.n_qubits = new_n_qubits

        # Then
        self.assertEqual(self.ansatz.n_qubits, new_n_qubits)

    def test_set_n_qubits_invalidates_circuits(self):
        # Given
        new_n_qubits = 100

        # When
        self.ansatz.n_qubits = new_n_qubits

        # Then
        self.assertIsNone(self.ansatz._circuit)
        self.assertIsNone(self.ansatz._gradient_circuits)

    def test_set_n_layers(self):
        # Given
        new_n_layers = 100

        # When
        self.ansatz.n_layers = new_n_layers

        # Then
        self.assertEqual(self.ansatz.n_layers, new_n_layers)

    def test_set_n_qubits_invalidates_circuits(self):
        # Given
        new_n_layers = 100

        # When
        self.ansatz.n_layers = new_n_layers

        # Then
        self.assertIsNone(self.ansatz._circuit)
        self.assertIsNone(self.ansatz._gradient_circuits)
