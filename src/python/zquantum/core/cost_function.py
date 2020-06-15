from .interfaces.cost_function import CostFunction
from .interfaces.backend import QuantumBackend
from .circuit import build_ansatz_circuit
from .measurement import ExpectationValues
from .utils import ValueEstimate
from openfermion import QubitOperator
from typing import Callable, Optional, Dict
import numpy as np
import importlib
import copy

from qeopenfermion import evaluate_qubit_operator
from zquantum.qaoa.ansatz import build_qaoa_circuit_grads
from openfermion import SymbolicOperator


class BasicCostFunction(CostFunction):
    """
    Basic implementation of the CostFunction interface.
    It allows to pass any function (and gradient) when initialized.

    Args:
        function (Callable): function we want to use as our cost function. Should take a numpy array as input and return a single number.
        gradient_function (Callable): function used to calculate gradients. Optional.
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        epsilon (float): epsilon used for calculating gradient using finite difference method.

    Params:
        function (Callable): see Args
        gradient_function (Callable): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        gradient_type (str): see Args
        save_evaluation_history (bool): see Args
        epsilon (float): see Args

    """

    def __init__(
        self,
        function: Callable,
        gradient_function: Optional[Callable] = None,
        gradient_type: str = "custom",
        save_evaluation_history: bool = True,
        epsilon: float = 1e-5,
    ):
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.gradient_type = gradient_type
        self.function = function
        self.gradient_function = gradient_function
        self.epsilon = epsilon

    def _evaluate(self, parameters: np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        value = self.function(parameters)
        return value

    def get_gradient(self, parameters: np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
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


class EvaluateOperatorCostFunction(CostFunction):
    """
    Cost function used for evaluating given operator using given ansatz.

    Args:
        target_operator (openfermion.QubitOperator): operator to be evaluated
        ansatz (dict): dictionary representing the ansatz
        backend (zquantum.core.interfaces.backend.QuantumBackend): backend used for evaluation
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        epsilon (float): epsilon used for calculating gradient using finite difference method.

    Params:
        target_operator (openfermion.QubitOperator): see Args
        ansatz (dict): see Args
        backend (zquantum.core.interfaces.backend.QuantumBackend): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        gradient_type (str): see Args
        epsilon (float): see Args

    """

    def __init__(
        self,
        target_operator: SymbolicOperator,
        ansatz: Dict,
        backend: QuantumBackend,
        gradient_type: str = "finite_difference",
        save_evaluation_history: bool = True,
        epsilon: float = 1e-5,
    ):
        self.target_operator = target_operator
        self.ansatz = ansatz
        self.backend = backend
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.gradient_type = gradient_type
        self.epsilon = epsilon

    def _evaluate(self, parameters: np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        circuit = build_ansatz_circuit(self.ansatz, parameters)
        expectation_values = self.backend.get_expectation_values(
            circuit, self.target_operator
        )
        final_value = np.sum(expectation_values.values)
        return final_value

    def get_gradient(self, parameters: np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        What method is used for calculating gradients is indicated by `self.gradient_type` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        if self.gradient_type == "finite_difference":
            return self.get_gradients_finite_difference(parameters)
        elif self.gradient_type == "qaoa":
            return self.get_gradients_qaoa(parameters)
        elif self.gradient_type == "schuld_shift":
            self.get_gradients_schuld_shift(parameters)
        else:
            raise Exception("Gradient type: %s is not supported", self.gradient_type)

    def get_gradients_qaoa(self, parameters):
        # Get circuits to measure gradient
        gradient_circuits, factors = build_qaoa_circuit_grads(
            parameters, self.ansatz["ansatz_kwargs"]["hamiltonians"]
        )

        # Run circuits to get expectation values for all gradient circuits wrt qubit operator
        expectation_lists = self._get_expectation_values_for_gradient_circuits_for_operator(
            gradient_circuits
        )

        # Get analytical gradient of operator wrt parameters from expectation values
        return self.get_analytical_gradient_from_expectation_values_for_operator(
            factors, expectation_lists
        )

    def get_gradients_schuld_shift(self, parameters):
        # Get circuits to measure gradient
        gradient_circuits, factors = gradient_circuits_for_simple_shift_rule(
            self, parameters
        )

        # Run circuits to get expectation values for all gradient circuits wrt qubit operator
        expectation_lists = self._get_expectation_values_for_gradient_circuits_for_operator(
            gradient_circuits
        )

        # Get analytical gradient of operator wrt parameters from expectation values
        return self.get_analytical_gradient_from_expectation_values_for_operator(
            factors, expectation_lists
        )

    def gradient_circuits_for_simple_shift_rule(self, params):
        """Construct a list of circuit corresponding to the 
        variational circuits that compute the contribution to the
        gradient, based on the shift rule (https://arxiv.org/abs/1811.11184).

        Args:
            ansatz (dict): the ansatz
            params (numpy.array): the ansatz parameters

        Returns:
            list_of_qprogs (list of lists of zmachine.core.circuit.Circuit: the circuits)
            factors (list of lists of floats): combination coefficients for the expectation
                values of the list of circuits.

        WARNING: This function applies to variational circuits where all the variational
        gates have independent parameters and generators with unique eigenvalues +1 and -1
        """
        factors = [1.0, -1.0]

        gradient_circuits = []
        output_factors = []
        module = importlib.import_module(self.ansatz["ansatz_module"])
        func = getattr(module, self.ansatz["ansatz_func"])

        for param_index in range(len(params)):

            circuits_per_param = []

            for factor in factors:

                new_ansatz_params = params.copy()
                new_ansatz_params[param_index] += factor * np.pi / 4.0
                circuits_per_param.append(
                    func(new_ansatz_params, **self.ansatz["ansatz_kwargs"])
                )

            gradient_circuits.append(circuits_per_param)
            output_factors.append(factors)

        return gradient_circuits, output_factors

    def _get_expectation_values_for_gradient_circuits_for_operator(
        self, gradient_circuits
    ):
        """ Computes a list of the expectation values of an operator with respect to
        a list of gradient circuits.

        Args:
            gradient_circuits (list of zmachine.core.circuit.Circuit): the circuits to run to measure the gradient
        
        Returns:
            list of zmachine.core.ExpectationValues objects

        WARNING: This function evaluates the gradient for ansatzes for which the function
            get_gradient_circuits_for_objective_function can be applied.
        """
        # Store expectation values in a list of lists of the same shape as gradients_circuits
        expectation_lists = []

        qubit_op_to_measure = QubitOperator()
        for term in self.target_operator.terms:
            qubit_op_to_measure.terms[term] = 1.0

        all_circuits = []
        for shifted_circuits_per_param in gradient_circuits:
            for circuit in shifted_circuits_per_param:
                all_circuits.append(circuit)

        # Get exp vals
        expectation_values_set = self.backend.get_expectation_values_for_circuitset(
            all_circuits, qubit_op_to_measure
        )

        # Store expectation values in a list of list of lists of the same shape as gradients_circuits
        expectation_lists = []
        counter = 0
        for shifted_circuits_per_param in gradient_circuits:
            param_list = []
            for circuit in shifted_circuits_per_param:
                param_list.append(expectation_values_set[counter])
                counter += 1
            expectation_lists.append(param_list)

        return expectation_lists

    def get_analytical_gradient_from_expectation_values_for_operator(
        self, factors, expectation_lists
    ):
        """Computes the analytical gradient vector for the given operator from provided lists
        of expectation values.

        Args:
            factors (list of lists): combination factors for the output of the gradient circuits
                for each param (size: n_params x n_circuits_per_params).
            expectation_lists: Nested list of expectation values as numpy arrays. Exact format
                depends on single_stateprep, If single_stateprep is False,
                expectation_lists[i][j][k] is a numpy array containing the expectation values 
                for the kth shifted circuit of the jth frame for the gradient of the ith parameter.
                If single_stateprep is True,
                expectation_lists[i][j] is a numpy array containing the expectation values for the
                jth shifted circuit for the gradient of the ith parameter.

        Returns:
            gradient (numpy array): The gradient of the objective function for each parameter.
        """

        gradient = np.zeros(len(expectation_lists))

        for i, shifted_exval_list in enumerate(expectation_lists):
            expectation_values = 0.0
            for j, exval in enumerate(shifted_exval_list):
                expectation_values += factors[i][j] * np.asarray(exval.values)
            gradient[i] = evaluate_qubit_operator(
                self.target_operator, ExpectationValues(expectation_values)
            ).value

        return gradient
