import pytest
import numpy as np
import sympy

from ._builtin_gates import (
    X,
    Y,
    Z,
    H,
    I,
    RX,
    RY,
    RZ,
    PHASE,
    T,
    CNOT,
    CZ,
    SWAP,
    ISWAP,
    CPHASE,
    XX,
    YY,
    ZZ,
)
from ._gates import Circuit, CustomGateDefinition


RNG = np.random.default_rng(42)

EXAMPLE_OPERATIONS = tuple(
    [
        *[gate(qubit_i) for qubit_i in [0, 1, 4] for gate in [X, Y, Z, H, I, T]],
        *[
            gate(angle)(qubit_i)
            for qubit_i in [0, 1, 4]
            for gate in [PHASE, RX, RY, RZ]
            for angle in [0, 0.1, np.pi / 5, np.pi, 2 * np.pi, sympy.Symbol("theta")]
        ],
        *[
            gate(qubit1_i, qubit2_i)
            for qubit1_i, qubit2_i in [(0, 1), (1, 0), (0, 5), (4, 2)]
            for gate in [CNOT, CZ, SWAP, ISWAP]
        ],
        *[
            gate(angle)(qubit1_i, qubit2_i)
            for qubit1_i, qubit2_i in [(0, 1), (1, 0), (0, 5), (4, 2)]
            for gate in [CPHASE, XX, YY, ZZ]
            for angle in [0, 0.1, np.pi / 5, np.pi, 2 * np.pi, sympy.Symbol("theta")]
        ],
    ]
)


def test_creating_circuit_has_correct_operations():
    circuit = Circuit(operations=EXAMPLE_OPERATIONS)
    assert circuit.operations == list(EXAMPLE_OPERATIONS)


class TestConcatenation:
    def test_appending_to_circuit_yields_correct_operations(self):
        circuit = Circuit()
        circuit += H(0)
        circuit += CNOT(0, 2)

        assert circuit.operations == [H(0), CNOT(0, 2)]
        assert circuit.n_qubits == 3

    def test_circuits_sum_yields_correct_operations(self):
        circuit1 = Circuit()
        circuit1 += H(0)
        circuit1 += CNOT(0, 2)

        circuit2 = Circuit([X(2), YY(sympy.Symbol("theta"))(5)])

        res_circuit = circuit1 + circuit2
        assert res_circuit.operations == [
            H(0),
            CNOT(0, 2),
            X(2),
            YY(sympy.Symbol("theta"))(5),
        ]
        assert res_circuit.n_qubits == 6


class TestBindingParams:
    def test_circuit_bound_with_all_params_contains_bound_gates(self):
        theta1, theta2, theta3 = sympy.symbols("theta1:4")
        symbols_map = {theta1: 0.5, theta2: 3.14, theta3: 0}

        circuit = Circuit(
            [
                RX(theta1)(0),
                RY(theta2)(1),
                RZ(theta3)(0),
                RX(theta3)(0),
            ]
        )
        bound_circuit = circuit.bind(symbols_map)

        expected_circuit = Circuit(
            [
                RX(theta1).bind(symbols_map)(0),
                RY(theta2).bind(symbols_map)(1),
                RZ(theta3).bind(symbols_map)(0),
                RX(theta3).bind(symbols_map)(0),
            ]
        )

        assert bound_circuit == expected_circuit

    def test_binding_all_params_leaves_no_free_symbols(self):
        alpha, beta, gamma = sympy.symbols("alpha,beta,gamma")
        circuit = Circuit(
            [
                RX(alpha)(0),
                RY(beta)(1),
                RZ(gamma)(0),
                RX(gamma)(0),
            ]
        )
        bound_circuit = circuit.bind({alpha: 0.5, beta: 3.14, gamma: 0})

        assert not bound_circuit.free_symbols

    def test_binding_some_params_leaves_free_params(self):
        theta1, theta2, theta3 = sympy.symbols("theta1:4")
        circuit = Circuit(
            [
                RX(theta1)(0),
                RY(theta2)(1),
                RZ(theta3)(0),
                RX(theta2)(0),
            ]
        )

        bound_circuit = circuit.bind({theta1: 0.5, theta3: 3.14})
        assert bound_circuit.free_symbols == {theta2}

    def test_binding_excessive_params_binds_only_the_existing_ones(self):
        theta1, theta2, theta3 = sympy.symbols("theta1:4")
        other_param = sympy.symbols("other_param")
        circuit = Circuit(
            [
                RX(theta1)(0),
                RY(theta2)(1),
                RZ(theta3)(0),
                RX(theta2)(0),
            ]
        )

        bound_circuit = circuit.bind({theta1: -np.pi, other_param: 42})
        assert bound_circuit.free_symbols == {theta2, theta3}


ALPHA = sympy.Symbol("alpha")
GAMMA = sympy.Symbol("gamma")
THETA = sympy.Symbol("theta")


CUSTOM_U_GATE = CustomGateDefinition(
    "U",
    sympy.Matrix([
        [THETA, GAMMA],
        [-GAMMA, THETA],
    ]),
    (THETA, GAMMA)
)


@pytest.mark.parametrize(
    "circuit,dict_",
    [
        (
            Circuit(),
            {
                "schema": "zapata-v1-circuit",
                "n_qubits": 0,
            },
        ),
        (
            Circuit([X(0)]),
            {
                "schema": "zapata-v1-circuit",
                "operations": [
                    {
                        "gate": {
                            "name": "X",
                        },
                        "qubit_indices": [0],
                    }
                ],
                "n_qubits": 1,
            },
        ),
        (
            Circuit([X(2), Y(1)]),
            {
                "schema": "zapata-v1-circuit",
                "operations": [
                    {
                        "gate": {
                            "name": "X",
                        },
                        "qubit_indices": [2],
                    },
                    {
                        "gate": {
                            "name": "Y",
                        },
                        "qubit_indices": [1],
                    },
                ],
                "n_qubits": 3,
            },
        ),
        (
            Circuit(
                [
                    H(0),
                    CNOT(0, 1),
                    RX(0)(5),
                ]
            ),
            {
                "schema": "zapata-v1-circuit",
                "operations": [
                    {
                        "gate": {
                            "name": "H",
                        },
                        "qubit_indices": [0],
                    },
                    {
                        "gate": {
                            "name": "CNOT",
                        },
                        "qubit_indices": [0, 1],
                    },
                    {
                        "gate": {
                            "name": "RX",
                            "params": ["0"],
                        },
                        "qubit_indices": [5],
                    },
                ],
                "n_qubits": 6,
            },
        ),
        (
            Circuit(
                operations=[
                    T(0),
                    CUSTOM_U_GATE(1, -1)(3),
                    CUSTOM_U_GATE(ALPHA, -1)(2),
                ],
                custom_gate_definitions=[CUSTOM_U_GATE],
            ),
            {
                "schema": "zapata-v1-circuit",
                "operations": [
                    {
                        "gate": {
                            "name": "T",
                        },
                        "qubit_indices": [0],
                    },
                    {
                        "gate": {
                            "name": "U",
                            "params": ["1", "-1"],
                        },
                        "qubit_indices": [3],
                    },
                    {
                        "gate": {
                            "name": "U",
                            "params": ["alpha", "-1"],
                        },
                        "qubit_indices": [2],
                    }
                ],
                "n_qubits": 4,
                "custom_gate_definitions": {
                    "U": [
                        ["theta", "gamma"],
                        ["-gamma", "theta"],
                    ]
                },
                "free_symbols": ["alpha"],
            },
        )
    ],
)
class TestSerialization:
    def test_serialized_dict_has_expected_form(self, circuit, dict_):
        assert circuit.to_dict() == dict_
