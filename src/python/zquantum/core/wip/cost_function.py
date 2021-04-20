from ..interfaces.backend import QuantumBackend
from ..interfaces.ansatz import Ansatz
from .estimators.estimation_interface import (
    EstimateExpectationValues,
    EstimationTaskTransformer,
    EstimationTask,
)
from .estimators.estimation import (
    naively_estimate_expectation_values,
    evaluate_circuits,
)
from ..interfaces.functions import function_with_gradient, StoreArtifact
from ..circuit import combine_ansatz_params, Circuit
from ..gradients import finite_differences_gradient
from ..utils import create_symbols_map, ValueEstimate
from ..measurement import ExpectationValues
from typing import Optional, Callable, Dict, List
import numpy as np
from openfermion import SymbolicOperator


def get_ground_state_cost_function(
    target_operator: SymbolicOperator,
    parametrized_circuit: Circuit,
    backend: QuantumBackend,
    estimator: EstimateExpectationValues = naively_estimate_expectation_values,
    estimation_tasks_transformations: List[EstimationTaskTransformer] = None,
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
        target_operator: operator to be evaluated and find the ground state of
        parametrized_circuit: parameterized circuit to prepare quantum states
        backend: backend used for evaluation
        estimator: estimator used to compute expectation value of target operator
        estimation_tasks_transformations: A list of callable functions that adhere to the EstimationTaskTransformer
            protocol and are used to create the estimation tasks.
        fixed_parameters: values for the circuit parameters that should be fixed.
        parameter_precision: the standard deviation of the Gaussian noise to add to each parameter, if any.
        parameter_precision_seed: seed for randomly generating parameter deviation if using parameter_precision
        gradient_function: a function which returns a function used to compute the gradient of the cost function
            (see from zquantum.core.gradients.finite_differences_gradient for reference)

    Returns:
        Callable
    """
    estimation_tasks = [
        EstimationTask(
            operator=target_operator, circuit=parametrized_circuit, number_of_shots=None
        )
    ]

    if estimation_tasks_transformations is None:
        estimation_tasks_transformations = []

    for transformer in estimation_tasks_transformations:
        estimation_tasks = transformer(estimation_tasks)

    circuit_symbols = sorted(
        list(
            set(
                [
                    param
                    for task in estimation_tasks
                    for param in task.circuit.symbolic_params
                ]
            )
        ),
        key=str,
    )

    def ground_state_cost_function(
        parameters: np.ndarray, store_artifact: StoreArtifact = None
    ) -> ValueEstimate:
        """Evaluates the expectation value of the op

        Args:
            parameters: parameters for the parameterized quantum circuit

        Returns:
            value: estimated energy of the target operator with respect to the circuit
        """
        nonlocal estimation_tasks
        parameters = parameters.copy()
        if fixed_parameters is not None:
            parameters = combine_ansatz_params(fixed_parameters, parameters)
        if parameter_precision is not None:
            rng = np.random.default_rng(parameter_precision_seed)
            noise_array = rng.normal(0.0, parameter_precision, len(parameters))
            parameters += noise_array

        symbols_map = create_symbols_map(circuit_symbols, parameters)
        estimation_tasks = evaluate_circuits(
            estimation_tasks, [symbols_map for _ in estimation_tasks]
        )

        expectation_values = estimator(backend, estimation_tasks)

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
        estimator_variance = 0
        for frame_covariance in expectation_values.estimator_covariances:
            estimator_variance += np.sum(frame_covariance, (0, 1))
        precision = np.sqrt(estimator_variance)
    return ValueEstimate(value, precision)


class AnsatzBasedCostFunction:
    """Cost function used for evaluating given operator using given ansatz.

    Args:
        target_operator: operator to be evaluated
        ansatz: ansatz used to evaluate cost function
        backend: backend used for evaluation
        estimator: estimator used to compute expectation value of target operator
        estimation_tasks_transformations: A list of callable functions that adhere to the EstimationTaskTransformer
            protocol and are used to create the estimation tasks.
        fixed_parameters: values for the circuit parameters that should be fixed.
        parameter_precision: the standard deviation of the Gaussian noise to add to each parameter, if any.
        parameter_precision_seed: seed for randomly generating parameter deviation if using parameter_precision

    Params:
        backend: see Args
        estimator: see Args
        fixed_parameters (np.ndarray): see Args
        parameter_precision: see Args
        parameter_precision_seed: see Args
        estimation_tasks: A list of EstimationTask objects with circuits to run and operators to measure
        circuit_symbols: A list of all symbolic parameters used in any estimation task
    """

    def __init__(
        self,
        target_operator: SymbolicOperator,
        ansatz: Ansatz,
        backend: QuantumBackend,
        estimator: EstimateExpectationValues = naively_estimate_expectation_values,
        estimation_tasks_transformations: List[EstimationTaskTransformer] = None,
        fixed_parameters: Optional[np.ndarray] = None,
        parameter_precision: Optional[float] = None,
        parameter_precision_seed: Optional[int] = None,
    ):
        self.backend = backend
        self.fixed_parameters = fixed_parameters
        self.parameter_precision = parameter_precision
        self.parameter_precision_seed = parameter_precision_seed

        if estimator is None:
            self.estimator = naively_estimate_expectation_values
        else:
            self.estimator = estimator

        if estimation_tasks_transformations is None:
            estimation_tasks_transformations = []

        self.estimation_tasks = [
            EstimationTask(
                operator=target_operator,
                circuit=ansatz.parametrized_circuit,
                number_of_shots=None,
            )
        ]
        for transformer in estimation_tasks_transformations:
            self.estimation_tasks = transformer(self.estimation_tasks)

        self.circuit_symbols = sorted(
            list(
                set(
                    [
                        param
                        for task in self.estimation_tasks
                        for param in task.circuit.symbolic_params
                    ]
                )
            ),
            key=str,
        )

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

        symbols_map = create_symbols_map(self.circuit_symbols, full_parameters)
        estimation_tasks = evaluate_circuits(
            self.estimation_tasks, [symbols_map for _ in self.estimation_tasks]
        )
        expectation_values = self.estimator(self.backend, estimation_tasks)

        return sum_expectation_values(expectation_values)
