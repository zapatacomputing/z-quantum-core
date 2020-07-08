import unittest
import numpy as np
from .cost_function import BasicCostFunction, AnsatzBasedCostFunction
from .interfaces.mock_objects import MockQuantumSimulator, MockEstimator, MockAnsatz
from .interfaces.cost_function_test import CostFunctionTests
from .utils import ValueEstimate
from openfermion import QubitOperator


class TestBasicCostFunction(unittest.TestCase, CostFunctionTests):
    def setUp(self):
        # Setting up inherited tests
        function = np.sum
        cost_function = BasicCostFunction(function)
        self.cost_functions = [cost_function]
        self.params_sizes = [1]

    def test_evaluate(self):
        # Given
        function = np.sum
        params_1 = np.array([1, 2, 3])
        params_2 = np.array([1, 2, 3, 4])
        target_value_1 = ValueEstimate(6)
        target_value_2 = ValueEstimate(10)
        cost_function = BasicCostFunction(function)

        # When
        value_1 = cost_function.evaluate(params_1)
        value_2 = cost_function.evaluate(params_2)
        history = cost_function.evaluations_history

        # Then
        self.assertEqual(value_1, target_value_1)
        self.assertEqual(value_2, target_value_2)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["value"], target_value_1)
        np.testing.assert_array_equal(history[0]["params"], params_1)
        self.assertEqual(history[1]["value"], target_value_2)
        np.testing.assert_array_equal(history[1]["params"], params_2)

    def test_get_gradient(self):
        # Given
        function = np.sum

        def gradient_function(params):
            return np.ones(params.size)

        params_1 = np.array([1, 2, 3])
        params_2 = np.array([1, 2, 3, 4])
        target_gradient_value_1 = np.array([1, 1, 1])
        target_gradient_value_2 = np.array([1, 1, 1, 1])
        cost_function = BasicCostFunction(
            function, gradient_function=gradient_function, gradient_type="custom"
        )

        # When
        gradient_value_1 = cost_function.get_gradient(params_1)
        gradient_value_2 = cost_function.get_gradient(params_2)

        # Then
        np.testing.assert_array_equal(gradient_value_1, target_gradient_value_1)
        np.testing.assert_array_equal(gradient_value_2, target_gradient_value_2)

    def test_get_finite_difference_gradient(self):
        # Given
        function = np.sum
        params_1 = np.array([1, 2, 3])
        params_2 = np.array([1, 2, 3, 4])
        target_gradient_value_1 = np.array([1, 1, 1])
        target_gradient_value_2 = np.array([1, 1, 1, 1])
        cost_function = BasicCostFunction(function, gradient_type="finite_difference")

        # When
        gradient_value_1 = cost_function.get_gradient(params_1)
        gradient_value_2 = cost_function.get_gradient(params_2)

        # Then
        np.testing.assert_almost_equal(gradient_value_1, target_gradient_value_1)
        np.testing.assert_almost_equal(gradient_value_2, target_gradient_value_2)


class TestAnsatzBasedCostFunction(unittest.TestCase, CostFunctionTests):
    def setUp(self):
        target_operator = QubitOperator("Z0")
        ansatz = MockAnsatz(number_of_layers=1, problem_size=1)
        backend = MockQuantumSimulator()
        estimator = MockEstimator()
        self.single_term_op_cost_function = AnsatzBasedCostFunction(
            target_operator, ansatz, backend, estimator=estimator
        )

        # Setting up inherited tests
        self.cost_functions = [self.single_term_op_cost_function]
        self.params_sizes = [1]

    def test_evaluate(self):
        # Given
        params = np.array([1])

        # When
        value_1 = self.single_term_op_cost_function.evaluate(params)
        value_2 = self.single_term_op_cost_function.evaluate(params)
        history = self.single_term_op_cost_function.evaluations_history

        # Then
        self.assertGreaterEqual(value_1.value, 0)
        self.assertLessEqual(value_1.value, 1)
        self.assertGreaterEqual(value_2.value, 0)
        self.assertLessEqual(value_2.value, 1)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["value"], value_1)
        np.testing.assert_array_equal(history[0]["params"], params)
        self.assertEqual(history[1]["value"], value_2)
        np.testing.assert_array_equal(history[1]["params"], params)
