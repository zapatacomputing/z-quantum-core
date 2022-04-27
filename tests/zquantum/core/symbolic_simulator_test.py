################################################################################
# © Copyright 2021 Zapata Computing Inc.
################################################################################
import pytest
import sympy
from zquantum.core import circuits
from zquantum.core.interfaces.backend_test import (
    QuantumSimulatorGatesTest,
    QuantumSimulatorTests,
)
from zquantum.core.symbolic_simulator import SymbolicSimulator


@pytest.fixture
def backend():
    return SymbolicSimulator()


@pytest.fixture
def wf_simulator() -> SymbolicSimulator:
    return SymbolicSimulator()


class TestSymbolicSimulator(QuantumSimulatorTests):
    gates_list = [
        circuits.XX(sympy.Symbol("theta"))(2, 1),
        circuits.U3(
            sympy.Symbol("alpha"),
            sympy.Symbol("beta"),
            sympy.Symbol("gamma"),
        )(1),
    ]
    incorrect_bindings = [
        {},
        {
            "alpha": sympy.pi,
            "beta": sympy.pi,
        },
    ]
    correct_bindings = [
        dict(zip(gate.free_symbols, [sympy.pi] * len(gate.free_symbols)))
        for gate in gates_list
    ]

    @pytest.mark.parametrize(
        "gate, binding",
        list(zip(gates_list, incorrect_bindings)),
    )
    def test_cannot_sample_from_circuit_containing_free_symbols(
        self, wf_simulator, gate, binding
    ):

        circuit = circuits.Circuit([gate])
        with pytest.raises(ValueError):
            wf_simulator.run_circuit_and_measure(
                circuit, n_samples=1000, symbol_map=binding
            )

    @pytest.mark.parametrize(
        "gate, binding",
        list(zip(gates_list, correct_bindings)),
    )
    def test_passes_for_complete_bindings(self, wf_simulator, gate, binding):
        circuit = circuits.Circuit([gate])
        wf_simulator.run_circuit_and_measure(
            circuit, n_samples=1000, symbol_map=binding
        )


class TestSymbolicSimulatorGates(QuantumSimulatorGatesTest):
    pass
