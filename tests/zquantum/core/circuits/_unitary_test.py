import cirq
import numpy as np
import pytest
import sympy
from zquantum.core.circuits import (
    RX,
    RY,
    RZ,
    XX,
    XY,
    YY,
    Circuit,
    H,
    I,
    X,
    Y,
    Z,
    export_to_cirq,
)


class TestCreatingUnitaryFromCircuit:
    @pytest.mark.parametrize(
        "circuit",
        [
            # Identity gates in some test cases below are used so that comparable
            # Cirq circuits have the same number of qubits as Zquantum ones.
            Circuit([RX(np.pi / 5)(0)]),
            Circuit([RY(np.pi / 2)(0), RX(np.pi / 5)(0)]),
            Circuit([I(1), I(2), I(3), I(4), RX(np.pi / 5)(0), XX(0.1)(5, 0)]),
            Circuit(
                [
                    XY(np.pi).controlled(1)(3, 1, 4),
                    RZ(0.1 * np.pi).controlled(2)(0, 2, 1),
                ]
            ),
            Circuit([H(1), YY(0.1).controlled(1)(0, 1, 2), X(2), Y(3), Z(4)]),
        ],
    )
    def test_without_free_params_gives_the_same_result_as_cirq(self, circuit):
        zquantum_unitary = circuit.to_unitary()
        cirq_circuit = export_to_cirq(circuit)
        cirq_unitary = cirq.unitary(cirq_circuit)
        assert isinstance(
            zquantum_unitary, np.ndarray
        ), "Unitary constructed form non-parameterized circuit is not a numpy array."

        np.testing.assert_array_almost_equal(zquantum_unitary, cirq_unitary)

    def test_commutes_with_parameter_substitution(self):
        theta, gamma = sympy.symbols("theta, gamma")
        circuit = Circuit(
            [RX(theta / 2)(0), X(1), RY(gamma / 4).controlled(1)(1, 2), YY(0.1)(0, 4)]
        )

        symbols_map = {theta: 0.1, gamma: 0.5}
        parameterized_unitary = circuit.to_unitary()
        unitary = circuit.bind(symbols_map).to_unitary()

        np.testing.assert_array_almost_equal(
            np.array(parameterized_unitary.subs(symbols_map), dtype=complex), unitary
        )
