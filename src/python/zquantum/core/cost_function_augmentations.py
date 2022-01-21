"""Implementation of cost function augmentations.

The "augmentation" is a function returning function. Augmentations can add side effects
to the function invocation (like logging or recording history), or even modify the
return value.

Cost functions are augmented via `augment_cost_function`. Basic usage looks as follows

augmented_func = augment_cost_function(func, [function_logger(level=level)])

If the function to be augmented has gradient, separate augmentations can be applied
to the function and the gradient, e.g.:

augmented_func = augment_cost_function(
    func,
    cost_function_augmentations=[function_logger(level=logging.INFO)],
    gradient_augmentations=[function_logger(level=logging.DEBUG)]
)


In principle, any function mapping cost function to cost function can be used as
augmentation. The common pattern however is an augmentation that triggers a side
effect, possibly when some conditions are met, and otherwise leaves the augmented
function unchanged. The `function_logger` augmentations is an example of such
augmentation.

There is a shortcut for implementing augmentations like function_logger. The required
steps are:
- Create class inheriting ConditionalSideEffect.
- Implement _act method, that defines what the side effect is.
- Create a thin wrapper function that constructs actual augmentation.
Refer to `function_logger` below for an example of this process.
"""
import abc
import logging
from typing import Callable, Iterable, Optional

from zquantum.core.history.save_conditions import SaveCondition, always
from zquantum.core.interfaces.cost_function import (
    CallableWithGradient,
    CallableWithGradientStoringArtifacts,
    CostFunction,
)

FunctionAugmentation = Callable[[Callable], Callable]


class _AttributePropagatingWrapper:
    """A wrapper for handling attribute propagation.

    This is intended to for handling attribute propagation between an object
    implementing augmentation and the function being augmented.

    This wrapper propagates all attribute accesses and calls to the wrapped `function`.
    If the attribute is not present on the `function`, `additional_source_of_attributes`
    is tried. If that fails, usual AttributeError is raised.

    Attributes can be overrides via attribute_overrides. This is to ensure that some
    attributes of wrapper objects are accessible even if they are also defined in the
    wrapped objects.
    """

    def __init__(self, function, additional_attribute_source, attribute_overrides):
        # We use object.__setattr__ here to not trigger propagation mechanism
        # implemented below. In  particular, usual self._function = function would fail
        # because it would already try to find the attribute on self._function, which
        # is undefined.
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
        """Propagate call to the wrapped function.

        This is needed for this object to be callable.
        """
        return self._function(*args, **kwargs)

    def __getattr__(self, attr_name):
        """Get attribute dynamically.

        This dunder method is called only if attr_name attribute is not
        present on the object itself. This implementation tries to find
        such an attribute in the following sources (in that order):
        - attribute overrides
        - wrapped function
        - additional attribute sources

        Overriding __getattr__ is crucial for propagating attribute queries to the
        wrapped objects.
        """
        try:
            return self._attribute_overrides[attr_name]
        except KeyError:
            try:
                return getattr(self._function, attr_name)
            except AttributeError:
                return getattr(self._additional_attribute_source, attr_name)

    def __setattr__(self, attr_name, value):
        """Set attribute not defined in this object's attribute dictionary.

        This dunder method is called only if attr_name is not present on the
        object itself. This implementation sets value of this attribute on the
        wrapped function if it is already present here, and in the additional
        attribute source otherwise.

        Overriding __setattr__ is crucial for propagating attribute settings to the
        wrapped objects.
        """
        setattr(
            self._function
            if hasattr(self._function, attr_name)
            else self._additional_attribute_source,
            attr_name,
            value,
        )


def _augment_function(function, augmentations, attribute_overrides):
    """Augment a function, possibly overriding some attributes."""
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
    """Augment a function and its gradient.

    Args:
        cost_function: a function to be augmented.
        cost_function_augmentations: augmentations to be applied to cost_function
            itself, in left-to-right order.
        gradient_augmentations: augmentations to be applied to a gradient. If the
            cost_function has no `gradient` attribute, this argument is ignored.
    Returns:
        A function with all application applied to it. The returned object
        has all attributes present in cost_function. The same is true about
        gradient. The exact behaviour of the returned object depends on the
        augmentations used, and the order in which they were applied.
    """
    attribute_overrides = (
        {
            "gradient": _augment_function(
                cost_function.gradient, gradient_augmentations, {}
            )
        }
        if isinstance(cost_function, CallableWithGradient)
        or isinstance(cost_function, CallableWithGradientStoringArtifacts)
        else {}
    )

    augmented_cost_function = _augment_function(
        cost_function, cost_function_augmentations, attribute_overrides
    )

    return augmented_cost_function


class ConditionalSideEffect(abc.ABC):
    """Base class for implementing augmentations that trigger some side effect.

    ConditionalSideEffect implementations don't modify the return value of
    the wrapped function, but trigger some additional action, possibly
    only when some conditions are met.

    Args:
        function: a single argument function to be wrapped
        predicate: a function with signature (result, params, call_number) where
          function(params) = result and call_number indicates how many times this
          object has been called so far. The side effect is triggerred if and only
          if predicate(result, params, call_number) is True. By default, side
          effect is triggerred always.

    Attributes:
        function: function passed to the initializer
        predicate: predicate passed to the initializer
        call_number: number indicating how many calls have been made to this
          function so far.

    Notes:
        Since no synchronization is implemented, ConditionalSideEffects are inherently
        unsuitable for multi-threaded use.

        Concrete subclasses of ConditionalSideEffect need to implement _act method.
    """

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
        """Perform side effect defined by this object.

        Args:
            result: value of wrapped function.
            params; corresponding parameters (i.e. self.function(params) = result.
        """


class FunctionLogger(ConditionalSideEffect):
    """Side effect causing function call to be logged.

    Args:
        function: function to be wrapped.
        predicate: boolean function defining when the logging should happen. See
          ConditionalSideEffect for more detailed description.
        logger: logger to be used. If not provided, logger with __name__ name will
          be used.
        level: log level to be used for logging messages. The default is INFO.
        message: %-style template string for message to be printed. Message will be
          constructed by passing format arguments (call number, function value).
    """

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
    """Cost function augmentation adding logging.

    For description of parameters see FunctionLogger class above.
    """

    def _augment(function):
        return FunctionLogger(function, predicate, logger, level, message)

    return _augment
