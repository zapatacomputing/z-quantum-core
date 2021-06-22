from abc import abstractmethod
from typing import Iterable, Sequence, TypeVar

from typing_extensions import Protocol

OperationType = TypeVar("OperationType")


class DecompositionRule(Protocol[OperationType]):
    @abstractmethod
    def production(self, operation: OperationType) -> Iterable[OperationType]:
        """Decompose given operation.

        Implementers should make sure that the result of applying the decomposed
        sequence to any wavefunction is the same as after applying the original
        operation.
        """

    @abstractmethod
    def predicate(self, operation: OperationType) -> bool:
        """Determine if this rule should be applied to given operation."""


def decompose_operation(
    operation: Iterable[OperationType],
    decomposition_rules: Sequence[DecompositionRule[OperationType]],
):
    if not decomposition_rules:
        return [operation]

    current_rule, *remaining_rules = decomposition_rules

    new_operations_to_decompose = (
        current_rule.production(operation)
        if current_rule.predicate(operation)
        else [operation]
    )
    return [
        decomposed_op
        for op in new_operations_to_decompose
        for decomposed_op in decompose_operation(op, remaining_rules)
    ]


def decompose_operations(
    operations: Iterable[OperationType],
    decomposition_rules: Sequence[DecompositionRule[OperationType]],
):
    return [
        decomposed_op
        for op in operations
        for decomposed_op in decompose_operation(op, decomposition_rules)
    ]
