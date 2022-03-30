import numpy as np
import pytest
from zquantum.core.circuits import (
    RY,
    U3,
    Circuit,
    X,
    Y,
    apply_gate_to_qubits,
    create_layer_of_gates,
)
from zquantum.core.utils import compare_unitary


class TestCreateLayerOfGates:
    def test_create_layer_of_gates_not_parametrized(self):
        # Given
        number_of_qubits = 4
        target_circuit = Circuit()
        gate = X
        for i in range(number_of_qubits):
            target_circuit += gate(i)

        # When
        layer_of_x = create_layer_of_gates(number_of_qubits, gate)

        # Then
        assert layer_of_x == target_circuit

    def test_create_layer_of_gates_parametrized(self):
        # Given
        gate = RY
        n_qubits_list = [2, 3, 4, 10]

        for n_qubits in n_qubits_list:
            # Given
            params = np.array([x for x in range(n_qubits)]).reshape(-1, 1)
            target_circuit = Circuit()
            for i in range(n_qubits):
                target_circuit += RY(*params[i])(i)

            u_zquantum = target_circuit.to_unitary()

            # When
            circuit = create_layer_of_gates(n_qubits, gate, params)
            unitary = circuit.to_unitary()

            # Then
            assert compare_unitary(unitary, u_zquantum, tol=1e-10)

    def test_create_layer_of_gates_wrong_num_params(self):
        params = np.ones(3).reshape(-1, 1)
        n_qubits = 2
        with pytest.raises(AssertionError):
            _ = create_layer_of_gates(n_qubits, RY, params)


class TestApplyGateToQubits:
    @pytest.fixture
    def base_circuit(self):
        circuit = Circuit()

        circuit += X(0)
        circuit += X(2)
        circuit += RY(np.pi / 3)(1)

        return circuit

    def test_apply_gate_to_qubits_not_parameterized(self, base_circuit):
        test_gate = Y
        qubit_indices = (1, 5, 4, 3)

        should_circuit = base_circuit
        for qubit in qubit_indices:
            should_circuit += test_gate(qubit)
        should_unitary = should_circuit.to_unitary()

        is_circuit = apply_gate_to_qubits(base_circuit, qubit_indices, test_gate)
        is_unitary = is_circuit.to_unitary()

        assert compare_unitary(should_unitary, is_unitary, tol=1e-10)

    @pytest.mark.parametrize(
        "test_gate,parameters",
        [(RY, np.arange(3).reshape(-1, 1)), (U3, np.arange(3 * 3).reshape(-1, 3))],
    )
    def test_apply_gate_to_qubits_parameterized(
        self, base_circuit, test_gate, parameters
    ):
        qubit_indices = (0, 2, 6)

        should_circuit = base_circuit
        for qubit, parameter in zip(qubit_indices, parameters):
            should_circuit += test_gate(*parameter)(qubit)
        should_unitary = should_circuit.to_unitary()

        is_circuit = apply_gate_to_qubits(
            base_circuit, qubit_indices, test_gate, parameters
        )
        is_unitary = is_circuit.to_unitary()

        assert compare_unitary(should_unitary, is_unitary, tol=1e-10)

    def test_apply_gate_to_qubits_wrong_num_params(self, base_circuit):
        params = np.ones(3).reshape(-1, 1)
        n_qubits = (2, 3)
        with pytest.raises(AssertionError):
            _ = apply_gate_to_qubits(base_circuit, n_qubits, RY, params)

    def test_apply_gate_to_qubits_raises_warning_with_duplicates(self, base_circuit):
        params = np.ones(2).reshape(-1, 1)
        n_qubits = (2, 2, 3)

        with pytest.warns(UserWarning):
            _ = apply_gate_to_qubits(base_circuit, n_qubits, RY, params)
