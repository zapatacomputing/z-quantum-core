from math import log2
from typing import Any, Dict, List, Set, Union
from warnings import warn

import numpy as np
from sympy import Matrix, Symbol


def _is_number(possible_number):
    try:
        complex(possible_number)
        return True
    except Exception:
        return False


def _cast_sympy_matrix_to_numpy(sympy_matrix, complex=False):
    new_type = np.complex128 if complex else np.float64

    try:
        return np.array(sympy_matrix, dtype=new_type).flatten()
    except TypeError:
        return np.array(sympy_matrix, dtype=object).flatten()


def _get_next_number_with_same_hamming_weight(val):
    # Copied from:
    # http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    t = (val | (val - 1)) + 1
    return t | ((((t & -t) // (val & -val)) >> 1) - 1)


def _most_significant_set_bit(val):
    bin_string = bin(val)
    return len(bin_string) - 2


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
    def amplitudes(self) -> Union[np.ndarray, Matrix]:
        if self.free_symbols:
            return _cast_sympy_matrix_to_numpy(self._amplitude_vector, complex=True)

        return self._amplitude_vector

    @property
    def n_qubits(self):
        return int(log2(len(self)))

    @property
    def free_symbols(self) -> Set[Symbol]:
        return getattr(self._amplitude_vector, "free_symbols", set())

    @staticmethod
    def _check_sanity(arr: Union[Matrix, np.ndarray]):
        def _calculate_probability_of_ground_entries(arr: Matrix) -> np.float64:
            numbers = np.array(
                [elem for elem in arr if _is_number(elem)], dtype=np.complex128
            )
            return np.sum(np.abs(numbers) ** 2)

        probs_of_ground_entries = _calculate_probability_of_ground_entries(arr)

        if (
            isinstance(arr, Matrix)
            and not arr.free_symbols
            or isinstance(arr, np.ndarray)
        ):
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
        cast_wf = _cast_sympy_matrix_to_numpy(self._amplitude_vector, complex=True)
        return f"Wavefunction({cast_wf})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Wavefunction):
            return False

        return self._amplitude_vector == other._amplitude_vector

    @staticmethod
    def zero_state(n_qubits: int) -> "Wavefunction":
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

    @staticmethod
    def dicke_state(n_qubits: int, hamming_weight: int) -> "Wavefunction":
        initial_wf = Wavefunction.zero_state(n_qubits)

        if hamming_weight < 0 or not isinstance(hamming_weight, int):
            raise ValueError(f"Invalid hamming weight value. Got {hamming_weight}.")

        if hamming_weight > n_qubits:
            raise ValueError(
                f"Hamming weight larger than number of qubits. \
                    Got {hamming_weight}. Max can be {n_qubits}."
            )

        if hamming_weight == 0:
            return initial_wf
        else:
            del initial_wf

            # Get first value with hamming weight
            current_value = int("1" * hamming_weight, base=2)

            counter: int = 1
            indices: List[int] = [current_value]
            while True:
                current_value = _get_next_number_with_same_hamming_weight(current_value)
                if not _most_significant_set_bit(current_value) <= n_qubits:
                    break
                indices.append(current_value)
                counter += 1

            amplitude = 1 / np.sqrt(counter)
            wf = np.zeros(2 ** n_qubits, dtype=np.complex128)
            wf[indices] = amplitude

            return Wavefunction(wf)

    def bind(self, symbol_map: Dict[Symbol, Any]) -> "Wavefunction":
        if not self.free_symbols:
            return self

        result = self._amplitude_vector.subs(symbol_map)

        try:
            return type(self)(result)
        except ValueError:
            raise ValueError("Passed map results in a violation of probability unity.")

    def probabilities(self) -> np.ndarray:
        return np.array([abs(elem) ** 2 for elem in self._amplitude_vector])

    def get_outcome_probs(self) -> Dict[str, float]:
        values = [
            format(i, "0" + str(self.n_qubits) + "b")[::-1] for i in range(len(self))
        ]

        probs = self.probabilities()

        return dict(zip(values, probs))


def flip_wavefunction(wavefunction: Wavefunction):
    return Wavefunction(flip_amplitudes(wavefunction.amplitudes))


def flip_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    number_of_states = len(amplitudes)
    ordering = [
        _flip_bits(n, number_of_states.bit_length() - 1)
        for n in range(number_of_states)
    ]
    return np.array([amplitudes[i] for i in ordering])


def _flip_bits(n, num_bits):
    return int(bin(n)[2:].zfill(num_bits)[::-1], 2)
