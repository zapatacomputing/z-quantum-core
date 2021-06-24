from typing import Any, Optional

import numpy as np
from pyquil.wavefunction import Wavefunction
from zquantum.core.circuits import Circuit
from zquantum.core.circuits.layouts import CircuitConnectivity
from zquantum.core.interfaces.backend import QuantumSimulator, flip_wavefunction
from zquantum.core.measurement import Measurements, sample_from_wavefunction


class SymbolicSimulator(QuantumSimulator):
    """A simulator computing wavefunction by consecutive gate matrix multiplication."""

    def __init__(
        self,
        n_samples: Optional[int] = None,
        noise_model: Optional[Any] = None,
        device_connectivity: Optional[CircuitConnectivity] = None,
    ):
        super().__init__(n_samples, noise_model, device_connectivity)

    def run_circuit_and_measure(
        self, circuit: Circuit, n_samples: Optional[int] = None, **kwargs
    ) -> Measurements:
        """Run a circuit and measure a certain number of bitstrings

        Args:
            circuit: the circuit to prepare the state
            n_samples: the number of bitstrings to sample
        Returns:
            The measured bitstrings.
        """
        if circuit.free_symbols:
            raise ValueError("Cannot sample from circuit with symbolic parameters.")

        if n_samples is None:
            if self.n_samples is None:
                raise ValueError(
                    "n_samples needs to be specified either as backend attribute or "
                    "as a function argument."
                )
            else:
                n_samples = self.n_samples
        wavefunction = self.get_wavefunction(circuit)
        bitstrings = sample_from_wavefunction(wavefunction, n_samples)
        return Measurements(bitstrings)

    def get_wavefunction(self, circuit: Circuit, **kwargs) -> Wavefunction:
        if circuit.free_symbols:
            raise ValueError("Currently circuits with free symbols are not supported")

        super().get_wavefunction(circuit, **kwargs)
        state = np.zeros(2 ** circuit.n_qubits)
        state[0] = 1

        for operation in circuit.operations:
            state = operation.apply(state)

        return flip_wavefunction(Wavefunction(state))
