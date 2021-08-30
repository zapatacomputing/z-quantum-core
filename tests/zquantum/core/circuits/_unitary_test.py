import numpy as np
import pytest
import sympy
from zquantum.core.circuits import RX, RY, RZ, XX, XY, YY, Circuit, H, I, X, Y, Z


class TestCreatingUnitaryFromCircuit:
    @pytest.mark.parametrize(
        "circuit, expected_unitary_filename",
        [
            # Identity gates in some test cases below are used so that comparable
            # Cirq circuits have the same number of qubits as Zquantum ones.
            (Circuit([RX(np.pi / 5)(0)]), "rx_gate.txt"),
            (Circuit([RY(np.pi / 2)(0), RX(np.pi / 5)(0)]), "ry_rx_gate.txt"),
            (
                Circuit([I(1), I(2), I(3), I(4), RX(np.pi / 5)(0), XX(0.1)(5, 0)]),
                "rx_xx_gate.txt",
            ),
            (
                Circuit(
                    [
                        XY(np.pi).controlled(1)(3, 1, 4),
                        RZ(0.1 * np.pi).controlled(2)(0, 2, 1),
                    ]
                ),
                "xy_rz_gate.txt",
            ),
            (
                Circuit([H(1), YY(0.1).controlled(1)(0, 1, 2), X(2), Y(3), Z(4)]),
                "h_yy_x_y_z_gate.txt",
            ),
        ],
    )
    def test_without_free_params_gives_the_same_result_as_cirq(
        self, circuit, expected_unitary_filename
    ):
        zquantum_unitary = circuit.to_unitary()

        """
        Note: We decided to go with file-based approach after extracting cirq
        from z-quantum-core and not being able to use `export_to_cirq` anymore.
        """
        path_to_array = (
            "/".join(__file__.split("/")[:-1])
            + "/cirq_unitaries/"
            + expected_unitary_filename
        )
        cirq_unitary = np.loadtxt(path_to_array, dtype=np.complex128)

        assert isinstance(
            zquantum_unitary, np.ndarray
        ), "Unitary constructed from non-parameterized circuit is not a numpy array."

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
