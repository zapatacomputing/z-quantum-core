from .interfaces.backend import QuantumBackend
from .circuit import Circuit, combine_ansatz_params
from .measurement import ExpectationValues
from openfermion import QubitOperator
from typing import Callable, Dict, Optional
import numpy as np
import importlib

from .openfermion import evaluate_qubit_operator
from zquantum.qaoa.ansatz import build_qaoa_circuit_grads
from openfermion import SymbolicOperator

from .estimator import BasicEstimator
from .gradients import finite_differences_gradient
from .interfaces.ansatz import Ansatz
from .interfaces.estimator import Estimator
from .interfaces.functions import StoreArtifact, function_with_gradient
from .utils import ValueEstimate, create_symbols_map


def get_ground_state_cost_function(
    target_operator: SymbolicOperator,
    parametrized_circuit: Circuit,
    backend: QuantumBackend,
    estimator: Estimator = BasicEstimator(),
    estimator_kwargs: Optional[Dict] = None,
    fixed_parameters: Optional[np.ndarray] = None,
    parameter_precision: Optional[float] = None,
    parameter_precision_seed: Optional[int] = None,
    gradient_function: Callable = finite_differences_gradient,
):
    """Returns a function that returns the estimated expectation value of the input
    target operator with respect to the state prepared by the parameterized quantum
    circuit when evaluated to the input parameters. The function also has a .gradient
    method when returns the gradient with respect the input parameters.

    Args:
        target_operator (openfermion.SymbolicOperator): operator to be evaluated and find the ground state of
        parametrized_circuit (zquantum.core.circuit.Circuit): parameterized circuit to prepare quantum states
        backend (zquantum.core.interfaces.backend.QuantumBackend): backend used for evaluation
        estimator: (zquantum.core.interfaces.estimator.Estimator) = estimator used to compute expectation value of target operator
        estimator_kwargs (dict): kwargs required to run get_estimated_expectation_values method of the estimator.
        gradient_function (Callable): a function which returns a function used to compute the gradient of the cost function
            (see from zquantum.core.gradients.finite_differences_gradient for reference)
        fixed_parameters (np.ndarray): values for the circuit parameters that should be fixed.
        parameter_precision (float): the standard deviation of the Gaussian noise to add to each parameter, if any.
        parameter_precision_seed (int): seed for randomly generating parameter deviation if using parameter_precision

    Returns:
        Callable
    """
    circuit_symbols = list(parametrized_circuit.symbolic_params)

    def ground_state_cost_function(
        parameters: np.ndarray, store_artifact: StoreArtifact = None
    ) -> ValueEstimate:
        """Evaluates the expectation value of the op

        Args:
            parameters: parameters for the parameterized quantum circuit

        Returns:
            value: estimated energy of the target operator with respect to the circuit
        """
        parameters = parameters.copy()
        if fixed_parameters is not None:
            parameters = combine_ansatz_params(fixed_parameters, parameters)
        if parameter_precision is not None:
            rng = np.random.default_rng(parameter_precision_seed)
            noise_array = rng.normal(0.0, parameter_precision, len(parameters))
            parameters += noise_array

        symbols_map = create_symbols_map(circuit_symbols, parameters)
        circuit = parametrized_circuit.evaluate(symbols_map)

        expectation_values = estimator.get_estimated_expectation_values(
            backend,
            circuit,
            target_operator,
            n_samples=backend.n_samples,
            **(estimator_kwargs if isinstance(estimator_kwargs, Dict) else {})
        )

        return ValueEstimate(np.sum(expectation_values.values))

    return function_with_gradient(
        ground_state_cost_function, gradient_function(ground_state_cost_function)
    )


def sum_expectation_values(expectation_values: ExpectationValues) -> ValueEstimate:
    """Compute the sum of expectation values.

    If correlations are available, the precision of the sum is computed as

    \epsilon = \sqrt{\sum_k \sigma^2_k}

    where the sum runs over frames and \sigma^2_k is the estimated variance of
    the estimated contribution of frame k to the total. This is calculated as

    \sigma^2_k = \sum_{i,j} Cov(o_{k,i}, o_{k, j})

    where Cov(o_{k,i}, o_{k, j}) is the estimated covariance in the estimated
    expectation values of operators i and j of frame k.

    Args:
        expectation_values: The expectation values to sum.

    Returns:
        The value of the sum, including a precision if the expectation values
            included covariances.
    """

    value = np.sum(expectation_values.values)

    precision = None

    if expectation_values.estimator_covariances:
        estimator_variance = 0.0
        for frame_covariance in expectation_values.estimator_covariances:
            estimator_variance += np.sum(frame_covariance, (0, 1))  # type: ignore
        precision = np.sqrt(estimator_variance)
    return ValueEstimate(value, precision)


class AnsatzBasedCostFunction:
    """Cost function used for evaluating given operator using given ansatz.

    Args:
        target_operator (openfermion.SymbolicOperator): operator to be evaluated
        ansatz (zquantum.core.interfaces.ansatz.Ansatz): ansatz used to evaluate cost function
        backend (zquantum.core.interfaces.backend.QuantumBackend): backend used for evaluation
        estimator: (zquantum.core.interfaces.estimator.Estimator) = estimator used to compute expectation value of target operator
        n_samples (int): number of samples (i.e. measurements) to be used in the estimator.
        estimator_kwargs(dict): kwargs required to run get_estimated_expectation_values method of the estimator.
        fixed_parameters (np.ndarray): values for the circuit parameters that should be fixed.
        parameter_precision (float): the standard deviation of the Gaussian noise to add to each parameter, if any.
        parameter_precision_seed (int): seed for randomly generating parameter deviation if using parameter_precision

    Params:
        target_operator (openfermion.SymbolicOperator): see Args
        ansatz (zquantum.core.interfaces.ansatz.Ansatz): see Args
        backend (zquantum.core.interfaces.backend.QuantumBackend): see Args
        estimator (zquantum.core.interfaces.estimator.Estimator): see Args
        estimator_kwargs(dict): see Args
        delta (float): see Args
        fixed_parameters (np.ndarray): see Args
        parameter_precision (float): see Args
        parameter_precision_seed (int): see Args
    """

    def __init__(
        self,
        target_operator: SymbolicOperator,
        ansatz: Ansatz,
        backend: QuantumBackend,
        estimator: Estimator = None,
        n_samples: Optional[int] = None,
        estimator_kwargs: Optional[Dict] = None,
        fixed_parameters: Optional[np.ndarray] = None,
        parameter_precision: Optional[float] = None,
        parameter_precision_seed: Optional[int] = None,
    ):
        self.target_operator = target_operator
        self.ansatz = ansatz
        self.backend = backend
        if estimator is None:
            self.estimator = BasicEstimator()
        else:
            self.estimator = estimator
        self.n_samples = n_samples

        if estimator_kwargs is None:
            self.estimator_kwargs = {}
        else:
            self.estimator_kwargs = estimator_kwargs
        self.fixed_parameters = fixed_parameters
        self.parameter_precision = parameter_precision
        self.parameter_precision_seed = parameter_precision_seed

    def __call__(self, parameters: np.ndarray) -> ValueEstimate:
        """Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.

        Returns:
            value: cost function value for given parameters.
        """
        full_parameters = parameters.copy()
        if self.fixed_parameters is not None:
            full_parameters = combine_ansatz_params(self.fixed_parameters, parameters)
        if self.parameter_precision is not None:
            rng = np.random.default_rng(self.parameter_precision_seed)
            noise_array = rng.normal(
                0.0, self.parameter_precision, len(full_parameters)
            )
            full_parameters += noise_array

        circuit = self.ansatz.get_executable_circuit(full_parameters)
        expectation_values = self.estimator.get_estimated_expectation_values(
            self.backend,
            circuit,
            self.target_operator,
            n_samples=self.n_samples,
            **self.estimator_kwargs
        )

        return sum_expectation_values(expectation_values)

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
        expectation_lists = (
            self._get_expectation_values_for_gradient_circuits_for_operator(
                gradient_circuits
            )
        )

        # Get analytical gradient of operator wrt parameters from expectation values
        return self.get_analytical_gradient_from_expectation_values_for_operator(
            factors, expectation_lists
        )

    def get_gradients_schuld_shift(self, parameters):
        # Get circuits to measure gradient
        gradient_circuits, factors = self.gradient_circuits_for_simple_shift_rule(
            self, parameters
        )

        # Run circuits to get expectation values for all gradient circuits wrt qubit operator
        expectation_lists = (
            self._get_expectation_values_for_gradient_circuits_for_operator(
                gradient_circuits
            )
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
            list_of_qprogs (list of lists of zquantum.core.circuit.Circuit: the circuits)
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
        """Computes a list of the expectation values of an operator with respect to
        a list of gradient circuits.

        Args:
            gradient_circuits (list of zquantum.core.circuit.Circuit): the circuits to run to measure the gradient

        Returns:
            list of zquantum.core..measurement.ExpectationValues objects

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
