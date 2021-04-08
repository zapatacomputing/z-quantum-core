from ...circuit import Circuit
from ...interfaces.backend import QuantumBackend
from ...measurement import ExpectationValues
from openfermion import SymbolicOperator
from typing import List, Callable
from dataclasses import dataclass
from typing_extensions import Protocol


@dataclass
class EstimationProblem:
    """
    Data class defining an estimation problem.

    Args:
        operator: Operator for which we want to calculate the expectation values
        circuit: Circuit used for evaluating the operator
        constraints: Define constraints used in the estimation process,
            e.g. number of shots or target accuracy.
    """

    operator: SymbolicOperator  # TODO: good type?
    circuit: Circuit
    number_of_shots: int


class EstimationProblemTransformer(Protocol):
    """Protocol defining function which transforms a list of EstimationProblems
    into another list of EstimationProblems.
    """

    def __call__(
        self, estimation_problems: List[EstimationProblem], **kwargs
    ) -> List[EstimationProblem]:
        pass


class EstimateExpectationValues(Protocol):
    def __call__(
        self, backend: QuantumBackend, estimation_problems: List[EstimationProblem]
    ) -> ExpectationValues:
        pass
