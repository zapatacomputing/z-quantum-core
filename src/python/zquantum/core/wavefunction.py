from math import log2
from typing import Any, Dict, List, Union
from warnings import warn

import numpy as np
from sympy import Matrix, Symbol


def _is_number(possible_number):
    try:
        complex(possible_number)
        return True
    except Exception:
        return False


def _free_symbols(array_or_matrix):
    return getattr(array_or_matrix, "free_symbols", set())


def _cast_sympy_matrix_to_numpy(sympy_matrix, complex=False):
    new_type = np.complex128 if complex else np.float64

    try:
        return np.array(sympy_matrix, dtype=new_type).flatten()
    except TypeError:
        return np.array(sympy_matrix, dtype=object).flatten()


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

        try:
            self._amplitude_vector = np.asarray(amplitude_vector, dtype=complex)
        except TypeError:
            self._amplitude_vector = Matrix(amplitude_vector)

        self._check_sanity(self._amplitude_vector)

    @property
    def amplitudes(self):
        if _free_symbols(self._amplitude_vector):
            return _cast_sympy_matrix_to_numpy(self._amplitude_vector, complex=True)

        return self._amplitude_vector

    @property
    def n_qubits(self):
        return int(log2(len(self)))

    @property
    def free_symbols(self):
        return _free_symbols(self._amplitude_vector)

    @staticmethod
    def _check_sanity(arr: Matrix):
        """
        Cases to watch out for:
        #1 no free symbols --> must ensure unit probability
        #2 free symbols --> must ensure numbers do not exceed unit probability already
        """

        def _calculate_probability_of_ground_entries(arr: Matrix) -> np.float64:
            numbers = np.array(
                [elem for elem in arr if _is_number(elem)], dtype=np.complex128
            )
            return np.sum(np.abs(numbers) ** 2)

        probs_of_ground_entries = _calculate_probability_of_ground_entries(arr)
        if not _free_symbols(arr):
            if not np.isclose(probs_of_ground_entries, 1.0):
                raise ValueError("Vector does not result in a unit probability.")
        else:
            if probs_of_ground_entries > 1.0:
                raise ValueError(
                    "Ground entries in vector already exceeding probability of 1.0!"
                )

    def __len__(self) -> int:
        return len(self._amplitude_vector)

    def __iter__(self):
        return iter(self._amplitude_vector)

    def __getitem__(self, idx):
        return self._amplitude_vector[idx]

    def __setitem__(self, idx, val):
        old_val = self._amplitude_vector[idx]
        self._amplitude_vector[idx] = val

        try:
            self._check_sanity(self._amplitude_vector)
        except ValueError:
            self._amplitude_vector[idx] = old_val

            raise ValueError("This assignment violates probability unity.")

    def __str__(self) -> str:
        return self._amplitude_vector.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Wavefunction):
            return False

        return self._amplitude_vector == other._amplitude_vector

    @staticmethod
    def init_system(n_qubits: int) -> "Wavefunction":
        if not isinstance(n_qubits, int):
            warn(
                f"Non-integer value {n_qubits} passed as number of qubits! "
                "Will be cast to integer."
            )
            n_qubits = int(n_qubits)

        if n_qubits <= 0:
            raise ValueError(f"Invalid number of qubits in system. Got {n_qubits}.")

        np_arr = np.zeros(2 ** n_qubits, dtype=np.complex128)
        np_arr[0] = 1.0
        return Wavefunction(np_arr)

    def bind(self, symbol_map: Dict[Symbol, Any]) -> "Wavefunction":
        if not _free_symbols(self._amplitude_vector):
            return self

        result = self._amplitude_vector.subs(symbol_map)

        try:
            return type(self)(result)
        except ValueError:
            raise ValueError("Passed map results in a violation of probability unity.")

    def probabilities(self) -> np.ndarray:
        return np.array([abs(elem) ** 2 for elem in self._amplitude_vector])

    def get_outcome_probs(self) -> Dict[str, float]:
        values = [format(i, "0" + str(self.n_qubits) + "b") for i in range(len(self))]

        probs = self.probabilities()

        return dict(zip(values, probs))
