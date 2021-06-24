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
def wf_simulator():
    return SymbolicSimulator()


class TestSymbolicSimulator(QuantumSimulatorTests):

    def test_get_wavefunction_raises_if_circuit_contains_free_symbols(
        self, wf_simulator
    ):
        circuit = circuits.Circuit([circuits.RX(sympy.Symbol("theta"))(2)])
        with pytest.raises(ValueError):
            wf_simulator.get_wavefunction(circuit)

    def test_cannot_sample_from_circuit_containing_free_symbols(self, wf_simulator):
        circuit = circuits.Circuit([circuits.XX(sympy.Symbol("theta"))(2, 1)])
        with pytest.raises(ValueError):
            wf_simulator.run_circuit_and_measure(circuit)


class TestSymbolicSimulatorGates(QuantumSimulatorGatesTest):
    pass
