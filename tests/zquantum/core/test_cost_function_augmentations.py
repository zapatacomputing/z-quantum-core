import logging

import numpy as np
import pytest
from zquantum.core.cost_function_augmentations import (
    ConditionalSideEffect,
    augment_cost_function,
    function_logger,
)
from zquantum.core.history.save_conditions import always, every_nth


class Example:
    def __init__(self, function, label):
        self.function = function
        self.label = label

    def __call__(self, params):
        return self.function(params)


def augmentation_1(function):
    return Example(function, "example_1")


def augmentation_2(function):
    return Example(function, "example_2")


class AugmentationWithNumber:
    def __init__(self, function):
        self.number = 100
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def foo(x):
    return x ** 2


class Foo:
    def __init__(self):
        self.number = 10
        self.string = "test"

    def __call__(self, x):
        return x ** 2


class FooGradient:
    attr = "custom-attr"

    def __call__(self, *args, **kwargs):
        return np.array([1.0, 2.0, 3.0])


class FooWithGradient:
    def __init__(self):
        self.gradient = FooGradient()

    def __call__(self, x):
        return x ** 3


class ExampleSideEffect(ConditionalSideEffect):
    def _act(self, result, params):
        print(f"{self.call_number} {result} {params}")


def example_side_effect_augmentation(predicate):
    def _augment(function):
        return ExampleSideEffect(function, predicate)

    return _augment


class TestAugmentingCostFunction:
    def test_returns_the_same_function_if_no_augmentations_are_provided(self):
        assert augment_cost_function(foo) == foo

    def test_applies_augmentation_to_cost_function_in_correct_order(self):
        augmented_foo = augment_cost_function(foo, [augmentation_1, augmentation_2])

        assert augmented_foo.label == "example_2"
        assert augmented_foo.function.label == "example_1"
        assert augmented_foo.function.function == foo

    class TestQueryingAttributes:
        def test_cost_function_attributes_are_propagated(self):
            augmented_foo = augment_cost_function(
                Foo(), [augmentation_1, augmentation_2]
            )
            assert augmented_foo.number == 10
            assert augmented_foo.string == "test"

        def test_augmentation_attributes_override_cost_function_attributes(self):
            augmented_foo = augment_cost_function(Foo(), [AugmentationWithNumber])
            assert augmented_foo.number == 100
            assert augmented_foo.string == "test"

        def test_gradient_attributes_are_propagated(self):
            augmented_foo = augment_cost_function(
                FooWithGradient(),
                gradient_augmentations=[augmentation_1, augmentation_2],
            )

            assert augmented_foo.gradient.attr == "custom-attr"

    class TestSettingAttributes:
        def test_attribute_is_set_on_function_if_it_is_not_present_on_augmentation(
            self,
        ):
            cost_function = Foo()
            augmented_foo = augment_cost_function(cost_function, [augmentation_1])

            augmented_foo.number = 100

            assert augmented_foo.number == 100
            assert cost_function.number == 100

        def test_attribute_is_set_only_on_augmentation_if_it_is_already_present(self):
            cost_function = Foo()
            augmented_foo = augment_cost_function(
                cost_function, [AugmentationWithNumber]
            )

            augmented_foo.number = 1000

            assert augmented_foo.number == 1000
            assert cost_function.number == 10

        def test_attribute_is_set_on_gradient_if_it_is_not_present_on_its_augmentation(
            self,
        ):
            cost_function = FooWithGradient()
            augmented_foo = augment_cost_function(cost_function, [augmentation_1])

            augmented_foo.gradient.attr = "test"

            assert augmented_foo.gradient.attr == "test"
            assert cost_function.gradient.attr == "test"


class TestConditionalSideEffect:
    def test_passes_through_value_returned_by_wrapped_function(self):
        augmented_foo = augment_cost_function(
            foo, [example_side_effect_augmentation(always)]
        )

        assert augmented_foo(10) == foo(10)

    def test_fires_only_when_the_predicate_is_true(self, capsys):
        augmented_foo = augment_cost_function(
            foo, [example_side_effect_augmentation(every_nth(2))]
        )

        augmented_foo(3)
        augmented_foo(4)
        augmented_foo(10)
        augmented_foo(5)

        assert capsys.readouterr().out == "0 9 3\n2 100 10\n"


class TestLoggingAugmentation:
    @pytest.mark.parametrize("level", [logging.INFO, logging.WARNING, logging.ERROR])
    def test_logs_with_correct_level(self, level, caplog):
        augmented_foo = augment_cost_function(foo, [function_logger(level=level)])

        with caplog.at_level(level):
            augmented_foo(3)

        assert "Function called. Call number: 0, function value: 9" in caplog.text

    def test_uses_provided_logger(self, caplog):
        logger = logging.getLogger("my.logger")
        augmented_foo = augment_cost_function(
            foo, [function_logger(level=logging.WARNING, logger=logger)]
        )

        with caplog.at_level(logging.WARNING, "my.logger"):
            augmented_foo(4)

        assert "Function called. Call number: 0, function value: 16" in caplog.text
