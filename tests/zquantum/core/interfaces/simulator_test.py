################################################################################
# © Copyright 2021 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
from zquantum.core.circuits import CNOT, RX, RY, Circuit, MultiPhaseOperation, Operation
from zquantum.core.interfaces.backend import QuantumSimulator
from zquantum.core.symbolic_simulator import SymbolicSimulator


class SymbolicSimulatorWithNonSupportedOperations(SymbolicSimulator):
    def is_natively_supported(self, operation: Operation) -> bool:
        return super().is_natively_supported(operation) and operation.gate.name != "RX"


class SymbolicSimulatorWithDefaultSetOfSupportedOperations(SymbolicSimulator):
    def is_natively_supported(self, operation: Operation) -> bool:
        return QuantumSimulator.is_natively_supported(self, operation)


@pytest.mark.parametrize(
    "circuit",
    [
        Circuit([RY(0.5)(0), RX(1)(1), CNOT(0, 2), RX(np.pi)(2)]),
        Circuit([RX(1)(1), CNOT(0, 2), RX(np.pi)(2), RY(0.5)(0)]),
        Circuit([RX(1)(1), CNOT(0, 2), RX(np.pi)(2), RY(0.5)(0), RX(0.5)(0)]),
    ],
)
def test_quantum_simulator_switches_between_native_and_nonnative_modes_of_execution(
    circuit,
):
    simulator = SymbolicSimulatorWithNonSupportedOperations()
    reference_simulator = SymbolicSimulator()

    np.testing.assert_array_equal(
        simulator.get_wavefunction(circuit).amplitudes,
        reference_simulator.get_wavefunction(circuit).amplitudes,
    )


def test_by_default_only_gate_operations_are_supported():
    simulator = SymbolicSimulatorWithDefaultSetOfSupportedOperations()
    assert simulator.is_natively_supported(RX(np.pi / 2)(1))
    assert not simulator.is_natively_supported(
        MultiPhaseOperation((0.5, 0.2, 0.3, 0.1))
    )
