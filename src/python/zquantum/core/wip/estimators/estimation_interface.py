from dataclasses import dataclass
from typing import List

from openfermion import SymbolicOperator
from typing_extensions import Protocol

from ...circuit import Circuit
from ...interfaces.backend import QuantumBackend
from ...measurement import ExpectationValues


@dataclass(frozen=True)
class EstimationTask:
    """
    Data class defining an estimation problem.

    Args:
        operator: Operator for which we want to calculate the expectation values
        circuit: Circuit used for evaluating the operator
        constraints: Define constraints used in the estimation process,
            e.g. number of shots or target accuracy.
    """

    operator: SymbolicOperator
    circuit: Circuit
    number_of_shots: int


class EstimationTaskTransformer(Protocol):
    """Protocol defining function which transforms a list of EstimationTasks
    into another list of EstimationTasks.
    """

    def __call__(
        self, estimation_problems: List[EstimationTask], **kwargs
    ) -> List[EstimationTask]:
        pass


class EstimateExpectationValues(Protocol):
    """Protocol defining function estimates expectation values for a list of estimation tasks.."""

    def __call__(
        self, backend: QuantumBackend, estimation_problems: List[EstimationTask]
    ) -> ExpectationValues:
        pass
