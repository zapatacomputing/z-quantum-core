from math import log2
from typing import Any, Dict, List, Union

import numpy as np
from sympy import Abs, Matrix, Symbol


def _calculate_probability_of_ground_entries(arr: Matrix) -> np.float64:
    numbers = np.array([elem for elem in arr if elem.is_number], dtype=np.complex128)
    return np.sum(np.abs(numbers) ** 2)


class Wavefunction:
    """
    A simple wavefunction data structure that can
    be used to calculate amplitudes of quantum states.

    Args:
        amplitude_vector: the initial amplitudes of the system,
            can either be a NumPy ndarray or a SymPy Matrix
    """

    def __init__(
        self, amplitude_vector: Union[np.ndarray, Matrix, List[complex]]
    ) -> None:
        if bin(len(amplitude_vector)).count("1") != 1:
            raise ValueError(
                "Provided wavefunction does not have a size of a power of 2."
            )

        if isinstance(amplitude_vector, np.ndarray) or isinstance(
            amplitude_vector, list
        ):
            amplitude_vector = Matrix(amplitude_vector)

        self._check_sanity(amplitude_vector)

        self._wavefunction = amplitude_vector

    @property
    def amplitudes(self):
        return np.array(self._wavefunction, dtype=np.complex128).flatten()

    @staticmethod
    def _check_sanity(arr: Matrix):
        """
        Cases to watch out for:
        #1 no free symbols --> must ensure unit probability
        #2 free symbols --> must ensure numbers do not exceed unit probability already
        """
        probs_of_ground_entries = _calculate_probability_of_ground_entries(arr)
        if len(arr.free_symbols) == 0:
            if not np.isclose(probs_of_ground_entries, 1.0):
                raise ValueError("Vector does not result in a unit probability.")
        else:
            if probs_of_ground_entries > 1.0:
                raise ValueError(
                    "Ground entries in vector already exceeding probability of 1.0!"
                )

    def __len__(self) -> int:
        return len(self._wavefunction)

    def __iter__(self):
        self._i = 0
        return iter(self._wavefunction)

    def __next__(self):
        if self._i < len(self):
            self._i += 1
            return self[self._i]

        raise StopIteration

    def __getitem__(self, idx):
        return self._wavefunction[idx]

    def __setitem__(self, idx, val):
        old_val = self._wavefunction[idx]
        self._wavefunction[idx] = val

        try:
            self._check_sanity(self._wavefunction)
        except ValueError:
            self._wavefunction[idx] = old_val

            raise ValueError("This assignment violates probability unity.")

    def __str__(self) -> str:
        return self._wavefunction.__str__()

    @staticmethod
    def init_system(n_qubits: int) -> "Wavefunction":
        np_arr = np.zeros(2 ** n_qubits, dtype=np.complex128)
        np_arr[0] = 1.0
        return Wavefunction(np_arr)

    def bind(self, symbol_map: Dict[Symbol, Any], overwrite_symbols=True) -> Matrix:
        result = self._wavefunction.subs(symbol_map)

        try:
            self._check_sanity(result)
        except ValueError:
            raise ValueError("Passed map results in a violation of probability unity.")

        if overwrite_symbols:
            self._wavefunction = result

        return result

    def probabilities(self) -> Matrix:
        absoluted_wf = Abs(self._wavefunction)
        squared_wf = absoluted_wf.multiply_elementwise(absoluted_wf)
        return squared_wf

    def get_outcome_probs(self) -> Dict[str, float]:
        values = [
            format(i, "0" + str(int(log2(len(self)))) + "b") for i in range(len(self))
        ]

        probs = self.probabilities()

        return dict(zip(values, probs))
