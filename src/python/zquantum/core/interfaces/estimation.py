from dataclasses import dataclass
from typing import List, Optional

from openfermion import SymbolicOperator
from typing_extensions import Protocol

from ..circuits import Circuit
from ..measurement import ExpectationValues
from .backend import QuantumBackend


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
    number_of_shots: Optional[int]


class EstimationPreprocessor(Protocol):
    """Protocol defining function which transforms a list of EstimationTasks
    into another list of EstimationTasks.
    """

    def __call__(self, estimation_tasks: List[EstimationTask]) -> List[EstimationTask]:
        pass


class EstimateExpectationValues(Protocol):
    """Protocol defining a function that estimates expectation values for a list of
    estimation tasks. Implementations of this protocol should obey the following rules:
    1. Return one ExpectationValue for each EstimationTask.
    2. The order in which ExpectationValues are returned should match the order
       in which EstimationTasks were provided.
    3. Number of entries in each ExpectationValue is not restricted.
    4. Output ExpectationValue should include coefficients of the terms/operators.
    5. estimation_tasks can include tasks where operator consists of a constant term or
        contains a constant term. The implemented method should include the
        contributions of such constant terms in the return value.
    """

    def __call__(
        self, backend: QuantumBackend, estimation_tasks: List[EstimationTask]
    ) -> List[ExpectationValues]:
        pass
