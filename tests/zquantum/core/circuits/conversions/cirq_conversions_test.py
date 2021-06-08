import cirq
import numpy as np
import pytest
import sympy
from zquantum.core.circuits import _builtin_gates, _circuit, _gates
from zquantum.core.circuits.conversions.cirq_conversions import (
    export_to_cirq,
    import_from_cirq,
    make_rotation_factory,
)

# --------- gates ---------

EQUIVALENT_IDENTITY_GATES = [
    (_builtin_gates.I, cirq.I),
]

EQUIVALENT_NON_PARAMETRIC_GATES = [
    (_builtin_gates.X, cirq.X),
    (_builtin_gates.Y, cirq.Y),
    (_builtin_gates.Z, cirq.Z),
    (_builtin_gates.H, cirq.H),
    (_builtin_gates.S, cirq.S),
    (_builtin_gates.T, cirq.T),
    (_builtin_gates.CNOT, cirq.CNOT),
    (_builtin_gates.CZ, cirq.CZ),
    (_builtin_gates.SWAP, cirq.SWAP),
    (_builtin_gates.ISWAP, cirq.ISWAP),
]

EQUIVALENT_PARAMETRIC_GATES = [
    (zq_cls(theta), cirq_cls(theta))
    for zq_cls, cirq_cls in [
        (_builtin_gates.RX, cirq.rx),
        (_builtin_gates.RY, cirq.ry),
        (_builtin_gates.RZ, cirq.rz),
        (_builtin_gates.RH, make_rotation_factory(cirq.HPowGate, 0.0)),
        (_builtin_gates.PHASE, make_rotation_factory(cirq.ZPowGate)),
        (_builtin_gates.CPHASE, cirq.cphase),
        (_builtin_gates.XX, make_rotation_factory(cirq.XXPowGate, -0.5)),
        (_builtin_gates.YY, make_rotation_factory(cirq.YYPowGate, -0.5)),
        (_builtin_gates.ZZ, make_rotation_factory(cirq.ZZPowGate, -0.5)),
        (_builtin_gates.XY, make_rotation_factory(cirq.ISwapPowGate, 0.0)),
    ]
    for theta in [0, -1, np.pi / 5, 2 * np.pi, -np.pi, -np.pi / 2, -np.pi / 4]
]

EQUIVALENT_U3_GATES = [
    (
        _builtin_gates.U3(theta, phi, lambda_),
        cirq.circuits.qasm_output.QasmUGate(
            theta / np.pi, phi / np.pi, lambda_ / np.pi
        ),
    )
    for theta, phi, lambda_ in [
        (0, 0, 0),
        (0, np.pi, 0),
        (np.pi, 0, 0),
        (0, 0, np.pi),
        (np.pi / 2, np.pi / 2, np.pi / 2),
        (0.1 * np.pi, 0.5 * np.pi, 0.3 * np.pi),
        # Below example does not work. Although matrices are the same, the params stored
        # in U3 are different.
        # (4.1 * np.pi / 2, 2.5 * np.pi, 3 * np.pi)
    ]
]


@pytest.mark.parametrize(
    "zquantum_gate,cirq_gate",
    [
        *EQUIVALENT_IDENTITY_GATES,
        *EQUIVALENT_NON_PARAMETRIC_GATES,
        *EQUIVALENT_PARAMETRIC_GATES,
        *EQUIVALENT_U3_GATES,
    ],
)
class TestGateConversion:
    def test_matrices_of_corresponding_zquantum_and_cirq_gates_are_equal(
        self, zquantum_gate, cirq_gate
    ):
        zquantum_matrix = np.array(zquantum_gate.matrix).astype(np.complex128)
        np.testing.assert_allclose(zquantum_matrix, cirq.unitary(cirq_gate), atol=1e-7)

    def test_exporting_gate_to_cirq_gives_expected_gate(self, zquantum_gate, cirq_gate):
        assert export_to_cirq(zquantum_gate) == cirq_gate

    def test_importing_gate_from_cirq_gives_expected_gate(
        self, zquantum_gate, cirq_gate
    ):
        assert import_from_cirq(cirq_gate) == zquantum_gate


@pytest.mark.parametrize(
    "zquantum_gate,cirq_gate",
    [
        *EQUIVALENT_NON_PARAMETRIC_GATES,
        *EQUIVALENT_PARAMETRIC_GATES,
    ],
)
def test_importing_gate_in_power_form_gives_expected_gate(zquantum_gate, cirq_gate):
    pow_gate = cirq_gate ** 1.0
    if "PowGate" not in str(type(pow_gate)):
        raise TypeError(
            f"This test expects power gates. Generated {type(pow_gate)} instead"
        )

    assert import_from_cirq(pow_gate) == zquantum_gate


# circuits ---------


THETA = sympy.Symbol("theta")
GAMMA = sympy.Symbol("gamma")

EXAMPLE_PARAM_VALUES = {
    THETA: 0.3,
    GAMMA: -5,
}

lq = cirq.LineQubit

EQUIVALENT_CIRCUITS = [
    (
        _circuit.Circuit([_builtin_gates.X(0), _builtin_gates.Z(2)]),
        cirq.Circuit([cirq.X(lq(0)), cirq.Z(lq(2))]),
    ),
    (
        _circuit.Circuit([_builtin_gates.CNOT(0, 1)]),
        cirq.Circuit([cirq.CNOT(lq(0), lq(1))]),
    ),
    (
        _circuit.Circuit([_builtin_gates.RX(np.pi)(1)]),
        cirq.Circuit([cirq.rx(np.pi)(lq(1))]),
    ),
    (
        _circuit.Circuit([_builtin_gates.SWAP.controlled(1)(2, 0, 3)]),
        cirq.Circuit([cirq.SWAP.controlled(1)(lq(2), lq(0), lq(3))]),
    ),
    (
        _circuit.Circuit([_builtin_gates.Y.controlled(2)(4, 5, 2)]),
        cirq.Circuit([cirq.Y.controlled(2)(lq(4), lq(5), lq(2))]),
    ),
]


EQUIVALENT_PARAMETRIZED_CIRCUITS = [
    (
        _circuit.Circuit([_builtin_gates.RX(THETA)(1)]),
        cirq.Circuit([cirq.rx(THETA)(lq(1))]),
    ),
    (
        _circuit.Circuit([_builtin_gates.RX(THETA * GAMMA)(1)]),
        cirq.Circuit([cirq.rx(THETA * GAMMA)(lq(1))]),
    ),
]


class CustomGate(cirq.Gate):
    """Example of Cirq custom gate.

    Taken from: https://quantumai.google/cirq/custom_gates
    """

    def __init__(self, theta):
        super(CustomGate, self)
        self.theta = theta

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return (
            np.array(
                [
                    [np.cos(self.theta), np.sin(self.theta)],
                    [np.sin(self.theta), -np.cos(self.theta)],
                ]
            )
            / np.sqrt(2)
        )

    def _circuit_diagram_info_(self, args):
        return f"R({self.theta})"


CIRQ_ONLY_CIRCUITS_WITH_FREE_SYMBOLS = [
    cirq.Circuit([cirq.CCXPowGate(exponent=-0.1, global_shift=THETA)(*lq.range(3))])
]

CIRQ_ONLY_CIRCUITS_WITHOUT_FREE_SYMBOLS = [
    cirq.Circuit([cirq.ZPowGate(exponent=1.2, global_shift=0.1)(lq(0))]),
    cirq.Circuit([CustomGate(np.pi)(lq(0))]),
]

CIRQ_ONLY_CIRCUITS = (
    CIRQ_ONLY_CIRCUITS_WITH_FREE_SYMBOLS + CIRQ_ONLY_CIRCUITS_WITHOUT_FREE_SYMBOLS
)


UNSUPPORTED_CIRQ_CIRCUITS = [
    cirq.Circuit([CustomGate(GAMMA)(lq(1))]),
]


class TestExportingToCirq:
    @pytest.mark.parametrize("zquantum_circuit,cirq_circuit", EQUIVALENT_CIRCUITS)
    def test_exporting_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, cirq_circuit
    ):
        converted = export_to_cirq(zquantum_circuit)
        assert (
            converted == cirq_circuit
        ), f"Converted circuit:\n{converted}\n isn't equal to\n{cirq_circuit}"

    @pytest.mark.parametrize(
        "zquantum_circuit, cirq_circuit", EQUIVALENT_PARAMETRIZED_CIRCUITS
    )
    def test_exporting_and_binding_parametrized_circuit_results_in_equivalent_circuit(
        self, zquantum_circuit, cirq_circuit
    ):
        converted = export_to_cirq(zquantum_circuit)
        converted_bound = cirq.resolve_parameters(converted, EXAMPLE_PARAM_VALUES)
        ref_bound = cirq.resolve_parameters(cirq_circuit, EXAMPLE_PARAM_VALUES)
        assert (
            converted_bound == ref_bound
        ), f"Converted circuit:\n{converted_bound}\n isn't equal to\n{ref_bound}"

    @pytest.mark.parametrize(
        "zquantum_circuit, cirq_circuit", EQUIVALENT_PARAMETRIZED_CIRCUITS
    )
    def test_binding_and_exporting_parametrized_circuit_results_in_equivalent_circuit(
        self, zquantum_circuit, cirq_circuit
    ):
        bound = zquantum_circuit.bind(EXAMPLE_PARAM_VALUES)
        bound_converted = export_to_cirq(bound)
        ref_bound = cirq.resolve_parameters(
            cirq_circuit, {**EXAMPLE_PARAM_VALUES, sympy.pi: 3.14}
        )
        assert cirq.approx_eq(
            bound_converted, ref_bound
        ), f"Converted circuit:\n{bound_converted}\n isn't equal to\n{ref_bound}"

    def test_daggers_are_converted_to_inverses(self):
        # NOTE: We don't add this test case to EQUIVALENT_CIRCUITS, because
        # only Zquantum -> cirq conversion is supported.
        zquantum_circuit = _circuit.Circuit(
            [_builtin_gates.X.dagger(2), _builtin_gates.T.dagger(1)]
        )
        cirq_circuit = cirq.Circuit(
            [cirq.inverse(cirq.X)(lq(2)), cirq.inverse(cirq.T)(lq(1))]
        )
        converted = export_to_cirq(zquantum_circuit)

        assert converted == cirq_circuit, (
            f"Converted circuit:\n{converted}\n isn't equal " f"to\n{cirq_circuit}"
        )


def _is_a_builtin_gate(gate: _gates.Gate):
    try:
        _builtin_gates.builtin_gate_by_name(gate.name)
        return True
    except KeyError:
        return False


class TestImportingFromCirq:
    @pytest.mark.parametrize("zquantum_circuit, cirq_circuit", EQUIVALENT_CIRCUITS)
    def test_gives_equivalent_circuit(self, zquantum_circuit, cirq_circuit):
        imported = import_from_cirq(cirq_circuit)
        assert imported == zquantum_circuit

    @pytest.mark.parametrize("cirq_circuit", CIRQ_ONLY_CIRCUITS)
    def test_with_cirq_only_gates_returns_custom_gates(self, cirq_circuit):
        circuit = import_from_cirq(cirq_circuit)
        for operation in circuit.operations:
            assert not _is_a_builtin_gate(operation.gate)

    @pytest.mark.parametrize("cirq_circuit", CIRQ_ONLY_CIRCUITS_WITHOUT_FREE_SYMBOLS)
    def test_with_cirq_only_gates_yields_correct_unitary(self, cirq_circuit):
        circuit = import_from_cirq(cirq_circuit)
        assert np.allclose(circuit.to_unitary(), cirq.unitary(cirq_circuit))

    @pytest.mark.parametrize("cirq_circuit", UNSUPPORTED_CIRQ_CIRCUITS)
    def test_with_unsupported_gates_raises_not_implemented_error(self, cirq_circuit):
        with pytest.raises(NotImplementedError):
            import_from_cirq(cirq_circuit)
