"""Test cases for ansatz-related utilities."""
import unittest
from unittest import mock

import numpy as np
import numpy.testing
from zquantum.core.interfaces.ansatz_utils import (
    DynamicProperty,
    ansatz_property,
    combine_ansatz_params,
    invalidates_parametrized_circuit,
)


class PseudoAnsatz:
    n_layers = ansatz_property(name="n_layers")

    def __init__(self, n_layers):
        self.n_layers = n_layers
        self._parametrized_circuit = None

    @property
    def parametrized_circuit(self):
        if self._parametrized_circuit is None:
            self._parametrized_circuit = f"Circuit with {self.n_layers} layers"
        return self._parametrized_circuit

    @invalidates_parametrized_circuit
    def rotate(self):
        """Mock method that "alters" some characteristics of ansatz.

        Using this method should invalidate parametrized circuit.
        """


class DynamicPropertyTests(unittest.TestCase):
    def test_uses_default_value_if_not_overwritten(self):
        class MyCls:
            x = DynamicProperty(name="x", default_value=-15)

        obj = MyCls()
        self.assertEqual(obj.x, -15)

    def test_can_be_set_in_init(self):
        class MyCls:
            length = DynamicProperty(name="length")

            def __init__(self, length):
                self.length = length

        obj = MyCls(0.5)
        self.assertEqual(obj.length, 0.5)

    def test_values_are_instance_dependent(self):
        class MyCls:
            height = DynamicProperty(name="height")

        obj1 = MyCls()
        obj2 = MyCls()

        obj1.height = 15
        obj2.height = 30

        self.assertEqual(obj1.height, 15)
        self.assertEqual(obj2.height, 30)


class TestAnsatzProperty(unittest.TestCase):
    """Note that we don't really need an ansatz intance, we only need to check that
    _parametrized_circuit is set to None.
    """

    def test_setter_resets_parametrized_circuit(self):
        ansatz = PseudoAnsatz(n_layers=10)

        # Trigger initial computation of parametrized circuit
        self.assertEqual(ansatz.parametrized_circuit, "Circuit with 10 layers")

        # Change n_layers -> check if it recalculated.
        ansatz.n_layers = 20
        self.assertIsNone(ansatz._parametrized_circuit)
        self.assertEqual(ansatz.parametrized_circuit, "Circuit with 20 layers")


class InvalidatesParametrizedCircuitTest(unittest.TestCase):
    def test_resets_parametrized_circuit(self):
        ansatz = PseudoAnsatz(n_layers=10)

        # Trigger initial computation of parametrized circuit
        self.assertEqual(ansatz.parametrized_circuit, "Circuit with 10 layers")

        # Trigger circuit invalidation
        ansatz.rotate()

        self.assertIsNone(ansatz._parametrized_circuit)

    def test_forwards_arguments_to_underlying_methods(self):
        method_mock = mock.Mock()
        decorated_method = invalidates_parametrized_circuit(method_mock)
        ansatz = PseudoAnsatz(n_layers=10)

        # Mock calling a regular method. Notice that we need to pass self explicitly
        decorated_method(ansatz, 2.0, 1.0, x=100, label="test")

        # Check that arguments were passed to underlying method
        method_mock.assert_called_once_with(ansatz, 2.0, 1.0, x=100, label="test")


def test_combine_ansatz_params():
    params1 = np.array([1.0, 2.0])
    params2 = np.array([3.0, 4.0])
    target_params = np.array([1.0, 2.0, 3.0, 4.0])

    combined_params = combine_ansatz_params(params1, params2)

    np.testing.assert_array_equal(combined_params, target_params)
