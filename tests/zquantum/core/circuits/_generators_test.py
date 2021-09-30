import numpy as np
import pytest
from zquantum.core.circuits import RY, Circuit, X, create_layer_of_gates
from zquantum.core.utils import compare_unitary


def test_create_layer_of_gates_not_parametrized():
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


def test_create_layer_of_gates_parametrized():
    # Given
    gate = RY
    n_qubits_list = [2, 3, 4, 10]

    for n_qubits in n_qubits_list:
        # Given
        params = [x for x in range(n_qubits)]
        target_circuit = Circuit()
        for i in range(n_qubits):
            target_circuit += RY(params[i])(i)

        u_zquantum = target_circuit.to_unitary()

        # When
        circuit = create_layer_of_gates(n_qubits, gate, params)
        unitary = circuit.to_unitary()

        # Then
        assert compare_unitary(unitary, u_zquantum, tol=1e-10)


def test_create_layer_of_gates_wrong_num_params():
    params = np.ones(3)
    n_qubits = 2
    with pytest.raises(AssertionError):
        _ = create_layer_of_gates(n_qubits, RY, params)
