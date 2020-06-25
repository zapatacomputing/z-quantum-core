from .interfaces.cost_function import CostFunction
from .interfaces.backend import QuantumBackend
from .interfaces.estimator import Estimator
from .circuit import build_ansatz_circuit
from .estimator import BasicEstimator
from .utils import ValueEstimate
from typing import Callable, Optional, Dict
import numpy as np
import copy
from openfermion import SymbolicOperator


class BasicCostFunction(CostFunction):
    """Basic implementation of the CostFunction interface.
    It allows to pass any function (and gradient) when initialized.

    Args:
        function (Callable): function we want to use as our cost function. Should take a numpy array as input and return a single number.
        gradient_function (Callable): function used to calculate gradients. Optional.
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        finite_diff_step_size(float): the step size used in finite difference approximation.

    Params:
        function (Callable): see Args
        gradient_function (Callable): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        gradient_type (str): see Args
        save_evaluation_history (bool): see Args
        finite_diff_step_size (float): see Args

    """

    def __init__(
        self,
        function: Callable,
        gradient_function: Optional[Callable] = None,
        gradient_type: str = "custom",
        save_evaluation_history: bool = True,
        finite_diff_step_size: float = 1e-5,
    ):
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.gradient_type = gradient_type
        self.function = function
        self.gradient_function = gradient_function
        self.finite_diff_step_size = finite_diff_step_size

    def _evaluate(self, parameters: np.ndarray) -> ValueEstimate:
        """Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur

        Returns:
            value: cost function value for given parameters.
        """
        value = ValueEstimate(self.function(parameters))
        return value

    def get_gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Evaluates the gradient of the cost function for given parameters.
        What method is used for calculating gradients is indicated by `self.gradient_type` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        if self.gradient_type == "custom":
            if self.gradient_function is None:
                raise Exception("Gradient function has not been provided.")
            else:
                return self.gradient_function(parameters)
        elif self.gradient_type == "finite_difference":
            if self.gradient_function is not None:
                raise Warning(
                    "Using finite difference method for calculating gradient even though self.gradient_function is defined."
                )
            return self.get_gradients_finite_difference(parameters)
        else:
            raise Exception("Gradient type: %s is not supported", self.gradient_type)


class AnsatzBasedCostFunction(CostFunction):
    """Cost function used for evaluating given operator using given ansatz.

    Args:
        target_operator (openfermion.QubitOperator): operator to be evaluated
        ansatz (dict): dictionary representing the ansatz
        backend (zquantum.core.interfaces.backend.QuantumBackend): backend used for evaluation
        estimator: (zquantum.core.interfaces.estimator.Estimator) = estimator used to compute expectation value of target operator 
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        finite_diff_step_size(float): the step size used in finite difference approximation.
        n_samples (int): number of samples (i.e. measurements) to be used in the estimator. 
        epsilon (float): an additive/multiplicative error term. The cost function should be computed to within this error term. 
        delta (float): a confidence term. If theoretical upper bounds are known for the estimation technique, 
            the final estimate should be within the epsilon term, with probability 1 - delta.

    Params:
        target_operator (openfermion.QubitOperator): see Args
        ansatz (dict): see Args
        backend (zquantum.core.interfaces.backend.QuantumBackend): see Args
        estimator: (zquantum.core.interfaces.estimator.Estimator) = see Args 
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        gradient_type (str): see Args
        finite_diff_step_size (float): see Args
        n_samples (int): see Args
        epsilon (float): see Args
        delta (float): see Args
    """

    def __init__(
        self,
        target_operator: SymbolicOperator,
        ansatz: Dict,
        backend: QuantumBackend,
        estimator: Estimator = None,
        gradient_type: str = "finite_difference",
        save_evaluation_history: bool = True,
        finite_diff_step_size: float = 1e-5,
        n_samples: Optional[int] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ):
        self.target_operator = target_operator
        self.ansatz = ansatz
        self.backend = backend
        if estimator is None:
            self.estimator = BasicEstimator()
        else:
            self.estimator = estimator
        self.gradient_type = gradient_type
        self.save_evaluation_history = save_evaluation_history
        self.finite_diff_step_size = finite_diff_step_size
        self.evaluations_history = []
        self.n_samples = n_samples
        self.epsilon = epsilon
        self.delta = delta

    def _evaluate(self, parameters: np.ndarray) -> ValueEstimate:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.

        Returns:
            value: cost function value for given parameters.
        """
        circuit = build_ansatz_circuit(self.ansatz, parameters)
        expectation_values = self.estimator.get_estimated_expectation_values(
            self.backend,
            circuit,
            self.target_operator,
            n_samples=self.n_samples,
            epsilon=self.epsilon,
            delta=self.delta,
        )
        final_value = np.sum(expectation_values.values)
        return ValueEstimate(final_value)
