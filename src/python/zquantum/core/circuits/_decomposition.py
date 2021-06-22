from typing import Iterable, TypeVar

import cirq
from typing_extensions import Protocol

OperationType = TypeVar("OperationType")


class DecompositionRule(Protocol[OperationType]):
    def production(self, operation: OperationType) -> Iterable[OperationType]:
        raise NotImplementedError()

    def predicate(self, operation: OperationType) -> bool:
        raise NotImplementedError()


def decompose_operation(
    operation: Iterable[OperationType],
    decomposition_rules: Iterable[DecompositionRule[OperationType]],
):
    rules_iter = iter(decomposition_rules)
    try:
        current_rule = next(rules_iter)
        if not current_rule.predicate(operation):
            return [operation]
        return [
            decomposed_op
            for op in current_rule.production(operation)
            for decomposed_op in decompose_operation(op, rules_iter)
        ]
    except StopIteration:
        return [operation]


def decompose_operations(
    operations: Iterable[OperationType],
    decomposition_rules: Iterable[DecompositionRule[OperationType]],
):
    return [
        decomposed_op
        for op in operations
        for decomposed_op in decompose_operation(op, decomposition_rules)
    ]
