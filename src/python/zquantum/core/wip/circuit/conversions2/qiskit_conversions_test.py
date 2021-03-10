import sympy
import numpy as np
import qiskit
import qiskit.circuit.random
import pytest

from zquantum.core.wip.circuit.conversions2.qiskit_conversions import (
    export_to_qiskit,
    import_from_qiskit,
)
from zquantum.core.wip.circuit import _gates
from zquantum.core.wip.circuit import _builtin_gates


# --------- gates ---------


EQUIVALENT_NON_PARAMETRIC_GATES = [
    (_builtin_gates.X, qiskit.circuit.library.XGate()),
    (_builtin_gates.Y, qiskit.circuit.library.YGate()),
    (_builtin_gates.Z, qiskit.circuit.library.ZGate()),
    (_builtin_gates.H, qiskit.circuit.library.HGate()),
    (_builtin_gates.I, qiskit.circuit.library.IGate()),
    (_builtin_gates.T, qiskit.circuit.library.TGate()),
    (_builtin_gates.CNOT, qiskit.extensions.CXGate()),
    (_builtin_gates.CZ, qiskit.extensions.CZGate()),
    (_builtin_gates.SWAP, qiskit.extensions.SwapGate()),
    (_builtin_gates.ISWAP, qiskit.extensions.iSwapGate()),
]

EQUIVALENT_PARAMETRIC_GATES = [
    (zquantum_cls(theta), qiskit_cls(theta))
    for zquantum_cls, qiskit_cls in [
        (_builtin_gates.RX, qiskit.circuit.library.RXGate),
        (_builtin_gates.RY, qiskit.circuit.library.RYGate),
        (_builtin_gates.RZ, qiskit.circuit.library.RZGate),
        (_builtin_gates.PHASE, qiskit.circuit.library.PhaseGate),
        (_builtin_gates.CPHASE, qiskit.extensions.CPhaseGate),
        (_builtin_gates.XX, qiskit.extensions.RXXGate),
        (_builtin_gates.YY, qiskit.extensions.RYYGate),
        (_builtin_gates.ZZ, qiskit.extensions.RZZGate),
    ]
    for theta in [0, -1, np.pi / 5, 2 * np.pi]
]


TWO_QUBIT_SWAP_MATRIX = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)


def _fix_qubit_ordering(qiskit_matrix):
    """Import qiskit matrix to ZQuantum matrix convention.

    Qiskit uses different qubit ordering than we do.
    It causes multi-qubit matrices to look different on first sight."""
    if len(qiskit_matrix) == 2:
        return qiskit_matrix
    if len(qiskit_matrix) == 4:
        return TWO_QUBIT_SWAP_MATRIX @ qiskit_matrix @ TWO_QUBIT_SWAP_MATRIX
    else:
        raise ValueError(f"Unsupported matrix size: {len(qiskit_matrix)}")


class TestGateConversion:
    @pytest.mark.parametrize(
        "zquantum_gate,qiskit_gate",
        [
            *EQUIVALENT_NON_PARAMETRIC_GATES,
            *EQUIVALENT_PARAMETRIC_GATES,
        ],
    )
    def test_matrices_are_equal(self, zquantum_gate, qiskit_gate):
        zquantum_matrix = np.array(zquantum_gate.matrix).astype(np.complex128)
        qiskit_matrix = _fix_qubit_ordering(qiskit_gate.to_matrix())
        np.testing.assert_allclose(zquantum_matrix, qiskit_matrix)


# --------- circuits ---------

# NOTE: In Qiskit, 0 is the most significant qubit,
# whereas in ZQuantum, 0 is the least significant qubit.
# This is we need to flip the indices.
#
# See more at
# https://qiskit.org/documentation/tutorials/circuits/1_getting_started_with_qiskit.html#Visualize-Circuit


def _qiskit_circuit_with_u1_gates():
    qc = qiskit.QuantumCircuit(7)
    qc.u1(0.42, 2)
    qc.u1(QISKIT_THETA, 1)
    return qc


def _make_qiskit_circuit(n_qubits, commands):
    qc = qiskit.QuantumCircuit(n_qubits)
    for method_name, method_args in commands:
        method = getattr(qc, method_name)
        method(*method_args)
    return qc


SYMPY_THETA = sympy.Symbol("theta")
SYMPY_GAMMA = sympy.Symbol("gamma")
QISKIT_THETA = qiskit.circuit.Parameter("theta")
QISKIT_GAMMA = qiskit.circuit.Parameter("gamma")


EXAMPLE_PARAM_VALUES = {
    "gamma": 0.3,
    "theta": -5,
}


EQUIVALENT_CIRCUITS = [
    (
        _gates.Circuit(
            [
                _builtin_gates.X(0),
                _builtin_gates.Z(2),
            ],
            6,
        ),
        _make_qiskit_circuit(6, [
            ('x', (0,)),
            ('z', (2,)),
        ]),
    ),
    (
        _gates.Circuit(
            [
                _builtin_gates.CNOT(0, 1),
            ],
            4,
        ),
        _make_qiskit_circuit(4, [
            ('cnot', (0, 1)),
        ]),
    ),
    (
        _gates.Circuit(
            [
                _builtin_gates.RX(np.pi)(1),
            ],
            4,
        ),
        _make_qiskit_circuit(4, [
            ('rx', (np.pi, 1)),
        ]),
    ),
    (
        _gates.Circuit(
            [_builtin_gates.SWAP.controlled(1)(2, 0, 3)],
            5,
        ),
        _make_qiskit_circuit(5, [
            ('append', (qiskit.circuit.library.SwapGate().control(1), [2, 0, 3])),
        ]),
    ),
    (
        _gates.Circuit(
            [_builtin_gates.Y.controlled(2)(4, 5, 2)],
            6,
        ),
        _make_qiskit_circuit(6, [
            ('append', (qiskit.circuit.library.YGate().control(2), [4, 5, 2])),
        ]),
    ),
]


EQUIVALENT_PARAMETRIZED_CIRCUITS = [
    (
        _gates.Circuit(
            [
                _builtin_gates.RX(SYMPY_THETA)(1),
            ],
            4,
        ),
        _make_qiskit_circuit(4, [
            ('rx', (QISKIT_THETA, 1)),
        ]),
    ),
    (
        _gates.Circuit(
            [
                _builtin_gates.RX(SYMPY_THETA * SYMPY_GAMMA)(1),
            ],
            4,
        ),
        _make_qiskit_circuit(4, [
            ('rx', (QISKIT_THETA * QISKIT_GAMMA, 1)),
        ]),
    ),
]


FOREIGN_QISKIT_CIRCUITS = [
    _qiskit_circuit_with_u1_gates(),
]


def _draw_qiskit_circuit(circuit):
    return qiskit.visualization.circuit_drawer(circuit, output="text")


class TestExportingToQiskit:
    @pytest.mark.parametrize("zquantum_circuit, qiskit_circuit", EQUIVALENT_CIRCUITS)
    def test_exporting_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, qiskit_circuit
    ):
        converted = export_to_qiskit(zquantum_circuit)
        assert converted == qiskit_circuit, (
            f"Converted circuit:\n{_draw_qiskit_circuit(converted)}\n isn't equal "
            f"to\n{_draw_qiskit_circuit(qiskit_circuit)}"
        )

    @pytest.mark.parametrize(
        "zquantum_circuit, qiskit_circuit", EQUIVALENT_PARAMETRIZED_CIRCUITS
    )
    def test_exporting_parametrized_circuit_doesnt_change_symbol_names(
        self, zquantum_circuit, qiskit_circuit
    ):
        converted = export_to_qiskit(zquantum_circuit)
        converted_names = sorted(map(str, converted.parameters))
        initial_names = sorted(map(str, zquantum_circuit.free_symbols))
        assert converted_names == initial_names

    @pytest.mark.parametrize(
        "zquantum_circuit, qiskit_circuit", EQUIVALENT_PARAMETRIZED_CIRCUITS
    )
    def test_exporting_and_binding_parametrized_circuit_results_in_equivalent_circuit(
        self, zquantum_circuit, qiskit_circuit
    ):
        converted = export_to_qiskit(zquantum_circuit)
        converted_bound = converted.bind_parameters(
            {param: EXAMPLE_PARAM_VALUES[str(param)] for param in converted.parameters}
        )
        ref_bound = qiskit_circuit.bind_parameters(
            {
                param: EXAMPLE_PARAM_VALUES[str(param)]
                for param in qiskit_circuit.parameters
            }
        )
        assert converted_bound == ref_bound, (
            f"Converted circuit:\n{_draw_qiskit_circuit(converted_bound)}\n isn't equal "
            f"to\n{_draw_qiskit_circuit(ref_bound)}"
        )

    @pytest.mark.parametrize(
        "zquantum_circuit, qiskit_circuit", EQUIVALENT_PARAMETRIZED_CIRCUITS
    )
    def test_binding_and_exporting_parametrized_circuit_results_in_equivalent_circuit(
        self, zquantum_circuit, qiskit_circuit
    ):
        bound = zquantum_circuit.bind(
            {
                symbol: EXAMPLE_PARAM_VALUES[str(symbol)]
                for symbol in zquantum_circuit.free_symbols
            }
        )
        bound_converted = export_to_qiskit(bound)
        ref_bound = qiskit_circuit.bind_parameters(
            {
                param: EXAMPLE_PARAM_VALUES[str(param)]
                for param in qiskit_circuit.parameters
            }
        )
        assert bound_converted == ref_bound, (
            f"Converted circuit:\n{_draw_qiskit_circuit(bound_converted)}\n isn't equal "
            f"to\n{_draw_qiskit_circuit(ref_bound)}"
        )

    def test_converting_circuit_with_daggers_fails_explicitly(self):
        # NOTE: Qiskit doesn't natively support dagger gates
        zquantum_circuit = _gates.Circuit(
            [_builtin_gates.X.dagger(2), _builtin_gates.T.dagger(1)], 3
        )
        with pytest.raises(NotImplementedError):
            export_to_qiskit(zquantum_circuit)


class TestImportingFromQiskit:
    @pytest.mark.parametrize("zquantum_circuit, qiskit_circuit", EQUIVALENT_CIRCUITS)
    def test_importing_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, qiskit_circuit
    ):
        imported = import_from_qiskit(qiskit_circuit)
        assert imported == zquantum_circuit

    @pytest.mark.parametrize("qiskit_circuit", FOREIGN_QISKIT_CIRCUITS)
    def test_importing_circuit_with_unsupported_gates_raises(self, qiskit_circuit):
        with pytest.raises(NotImplementedError):
            import_from_qiskit(qiskit_circuit)
