import abc
import logging
from typing import Callable, Iterable, Optional

from zquantum.core.history.save_conditions import SaveCondition, always
from zquantum.core.interfaces.cost_function import CostFunction

FunctionAugmentation = Callable[[Callable], Callable]


class _AttributePropagatingWrapper:
    def __init__(self, function, additional_attribute_source, attribute_overrides):
        object.__setattr__(self, "_function", function)
        object.__setattr__(
            self, "_additional_attribute_source", additional_attribute_source
        )
        object.__setattr__(
            self,
            "_attribute_overrides",
            attribute_overrides if attribute_overrides else {},
        )

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    def __getattr__(self, attr_name):
        try:
            return self._attribute_overrides[attr_name]
        except KeyError:
            try:
                return getattr(self._function, attr_name)
            except AttributeError:
                return getattr(self._additional_attribute_source, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(
            self._function
            if hasattr(self._function, attr_name)
            else self._additional_attribute_source,
            attr_name,
            value,
        )


def _augment_function(function, augmentations, attribute_overrides):
    if not augmentations:
        return function
    augmented_function = function
    for augmentation in augmentations:
        augmented_function = _AttributePropagatingWrapper(
            augmentation(augmented_function), augmented_function, attribute_overrides
        )

    return augmented_function


def augment_cost_function(
    cost_function: CostFunction,
    cost_function_augmentations: Optional[Iterable[FunctionAugmentation]] = None,
    gradient_augmentations: Optional[Iterable[FunctionAugmentation]] = None,
):
    attribute_overrides = (
        {
            "gradient": _augment_function(
                cost_function.gradient, gradient_augmentations, {}
            )
        }
        if hasattr(cost_function, "gradient")
        else {}
    )

    augmented_cost_function = _augment_function(
        cost_function, cost_function_augmentations, attribute_overrides
    )

    return augmented_cost_function


class ConditionalSideEffect(abc.ABC):
    def __init__(self, function: Callable, predicate: SaveCondition = always):
        self.function = function
        self.predicate = predicate
        self.call_number = 0

    def __call__(self, params):
        result = self.function(params)
        if self.predicate(result, params, self.call_number):
            self._act(result, params)

        self.call_number += 1
        return result

    @abc.abstractmethod
    def _act(self, result, params):
        pass


class FunctionLogger(ConditionalSideEffect):
    def __init__(
        self,
        function: Callable,
        predicate: SaveCondition = always,
        logger=None,
        level=logging.INFO,
        message="Function called. Call number: %d, function value: %s",
    ):
        super().__init__(function, predicate)
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.level = level
        self.message = message

    def _act(self, result, params):
        self.logger.log(self.level, self.message, self.call_number, result)


def function_logger(
    predicate: SaveCondition = always,
    logger=None,
    level=logging.INFO,
    message="Function called. Call number: %d, function value: %s",
):
    def _augment(function):
        return FunctionLogger(function, predicate, logger, level, message)

    return _augment
