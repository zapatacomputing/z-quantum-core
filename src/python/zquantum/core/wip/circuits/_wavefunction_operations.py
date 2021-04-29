from dataclasses import dataclass

import sympy
from ._gates import _sub_symbols, Parameter
from typing import Tuple, Sequence
import numpy as np


@dataclass(frozen=True)
class MultiPhaseOperation:
    """ TODO """

    params: Tuple[Parameter, ...]

    def bind(self, symbols_map) -> "MultiPhaseOperation":
        return self.replace_params(
            tuple(_sub_symbols(param, symbols_map) for param in self.params)
        )

    def replace_params(
        self, new_params: Tuple[Parameter, ...]
    ) -> "MultiPhaseOperation":
        return MultiPhaseOperation(new_params)

    def apply(self, wavefunction: Sequence[Parameter]) -> Sequence[Parameter]:
        if len(wavefunction) != len(self.params):
            raise ValueError(
                f"MultiPhaseOperation with {len(self.params)} params cannot be applied to wavefunction of length {len(wavefunction)}."
            )

        try:
            exp_params = np.exp(np.asarray(self.params, dtype=float) * 1j)
        except TypeError as e:
            raise RuntimeError(
                "MultiPhaseOperation can only be used only if params are real numbers."
            ) from e
        return np.multiply(wavefunction, exp_params)
