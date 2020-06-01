import unittest
import numpy as np

class CostFunctionTests(object):
    # To run tests with this base class, the following variables need to be properly initialized in the child class:
        # self.cost_functions
        # self.params_sizes

    def test_evaluate_returns_number(self):
        for cost_function, num_params in zip(self.cost_functions, self.params_sizes):
            # Given
            params = np.random.rand(num_params)
            # When
            value = cost_function.evaluate(params)
            # Then
            # self.assertIsInstance(value, (int, float))

    def test_evaluatue_saves_evaluation_history(self):
        for cost_function, num_params in zip(self.cost_functions, self.params_sizes):
            # Given
            params = np.random.rand(num_params)
            cost_function.save_evaluation_history = True
            # When
            value = cost_function.evaluate(params)
            # Then
            self.assertEqual(len(cost_function.evaluations_history), 1)
            np.testing.assert_equal(cost_function.evaluations_history[0]["value"], value)
            np.testing.assert_equal(cost_function.evaluations_history[0]["params"], params)

    def test_evaluatue_does_not_save_evaluation_history(self):
        for cost_function, num_params in zip(self.cost_functions, self.params_sizes):
            # Given
            params = np.random.rand(num_params)
            cost_function.save_evaluation_history = False
            # When
            value = cost_function.evaluate(params)
            # Then
            self.assertEqual(len(cost_function.evaluations_history), 0)

    def test_get_gradient(self):
        for cost_function, num_params in zip(self.cost_functions, self.params_sizes):
            # Given
            params = np.random.rand(num_params)
            # When
            gradient = cost_function.get_gradient(params)
            # Then
            self.assertEqual(len(gradient), len(params))
            self.assertIsInstance(gradient, np.ndarray)

    def test_get_gradient_raises_error_with_undefined_gradient_type(self):
        for cost_function, num_params in zip(self.cost_functions, self.params_sizes):
            # Given
            params = np.random.rand(num_params)
            cost_function.gradient_type = "GRADIENT DOES NOT EXIST"
            # When/Then
            self.assertRaises(Exception, lambda: cost_function.get_gradient(params))

    def test_get_gradients_finite_difference(self):
        for cost_function, num_params in zip(self.cost_functions, self.params_sizes):
            # Given
            params = np.random.rand(num_params)
            # When
            gradient = cost_function.get_gradients_finite_difference(params)
            # Then
            self.assertEqual(len(gradient), len(params))
            self.assertIsInstance(gradient, np.ndarray)