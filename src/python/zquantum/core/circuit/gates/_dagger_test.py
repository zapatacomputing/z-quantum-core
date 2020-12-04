"""Test cases for dagger operator."""
import pytest
import sympy
from . import H, X, Y, Z, I, CustomGate, ControlledGate, CNOT, CZ, SWAP

THETA = sympy.Symbol("theta")
PHI = sympy.Symbol("phi")
EXAMPLE_CUSTOM_GATE = CustomGate(
    sympy.Matrix(
    [
    [THETA, 0, 0, 0],
    [0, THETA, 0, 0],
    [0, 0, 0, -1j * PHI],
    [0, 0, -1j * PHI, 0],
    ]
    ),
    (2, 3),
)


@pytest.mark.parametrize(
    "gate",
    [
        X(0),
        Y(1),
        Z(2),
        EXAMPLE_CUSTOM_GATE,
        ControlledGate(
            EXAMPLE_CUSTOM_GATE,
            0
        )
    ]
)
class TestBasicPropertiesOfDaggerOperations:

    def test_applying_dagger_twice_gives_gate_equivalent_to_the_original_gate(
        self, gate
    ):
        second_gate = gate.dagger.dagger
        assert second_gate == gate

    def test_applying_dagger_twice_preserves_type_of_the_original_gate(
        self, gate
    ):
        second_gate = gate.dagger.dagger
        assert type(second_gate) == type(gate)

    def test_applying_dagger_gives_gate_with_the_same_qubits_as_the_original_one(
        self, gate
    ):
        assert gate.dagger.qubits == gate.qubits


def test_applying_dagger_to_controlled_gate_gives_controlled_gate_of_target_gates_dagger():
    dagger = ControlledGate(EXAMPLE_CUSTOM_GATE, 0).dagger
    assert isinstance(dagger,  ControlledGate)
    assert dagger.target_gate == EXAMPLE_CUSTOM_GATE.dagger


@pytest.mark.parametrize("gate_cls", [X, Y, Z, H, I])
def test_dagger_of_hermitian_single_qubit_gates_is_the_same_as_the_original_gate(
    gate_cls
):
    gate = gate_cls(0)
    assert gate is gate.dagger  # Notice that this is stronger than equality


@pytest.mark.parametrize("gate_cls", [CNOT, CZ])
def test_dagger_of_hermitian_controlled_two_qubit_gates_is_the_same_as_original_gate(
    gate_cls
):
    gate = gate_cls(0, 1)

    assert gate is gate.dagger


def test_dagger_of_swap_gate_is_the_same_as_the_original_gate():
    swap = SWAP(0, 2)

    assert swap.dagger is swap
