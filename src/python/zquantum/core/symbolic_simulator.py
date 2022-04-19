################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import Any, Dict, Optional

from sympy import Symbol
from zquantum.core.circuits import Circuit, Operation
from zquantum.core.circuits.layouts import CircuitConnectivity
from zquantum.core.interfaces.backend import QuantumSimulator, StateVector
from zquantum.core.measurement import Measurements, sample_from_wavefunction


class SymbolicSimulator(QuantumSimulator):
    """A simulator computing wavefunction by consecutive gate matrix multiplication.

    Args:
        seed: the seed of the sampler
    """

    def __init__(
        self,
        noise_model: Optional[Any] = None,
        device_connectivity: Optional[CircuitConnectivity] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(noise_model, device_connectivity)
        self._seed = seed

    def run_circuit_and_measure(
        self,
        circuit: Circuit,
        n_samples: int,
        symbol_map: Optional[Dict[Symbol, Any]] = None,
    ) -> Measurements:
        """Run a circuit and measure a certain number of bitstrings

        Args:
            circuit: the circuit to prepare the state
            n_samples: the number of bitstrings to sample
        """
        wavefunction = self.get_wavefunction(
            circuit if symbol_map is None else circuit.bind(symbol_map)
        )

        if wavefunction.free_symbols:
            raise ValueError(
                "Cannot sample from wavefunction with symbolic parameters."
            )

        bitstrings = sample_from_wavefunction(wavefunction, n_samples, self._seed)
        return Measurements(bitstrings)

    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:
        state = initial_state

        for operation in circuit.operations:
            state = operation.apply(state)

        return state

    def is_natively_supported(self, operation: Operation) -> bool:
        return True
