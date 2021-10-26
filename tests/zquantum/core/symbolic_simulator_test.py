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
    @pytest.mark.parametrize(
        "circuit_list, binding",
        [
            ([circuits.XX(sympy.Symbol("theta"))(2, 1)], {}),
            (
                [
                    circuits.U3(
                        sympy.Symbol("alpha"),
                        sympy.Symbol("beta"),
                        sympy.Symbol("gamma"),
                    )(1)
                ],
                {
                    "alpha": sympy.pi,
                    "beta": sympy.pi,
                },
            ),
        ],
    )
    def test_cannot_sample_from_circuit_containing_free_symbols(
        self, wf_simulator, circuit_list, binding
    ):
        circuit = circuits.Circuit(circuit_list)
        with pytest.raises(ValueError):
            wf_simulator.run_circuit_and_measure(
                circuit, n_samples=1000, symbol_map=binding
            )


class TestSymbolicSimulatorGates(QuantumSimulatorGatesTest):
    pass
