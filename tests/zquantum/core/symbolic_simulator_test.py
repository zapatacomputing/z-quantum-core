import pytest
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
    pass


class TestSymbolicSimulatorGates(QuantumSimulatorGatesTest):
    pass
