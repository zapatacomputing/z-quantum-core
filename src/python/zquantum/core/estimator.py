from .interfaces.estimator import Estimator
from .interfaces.backend import QuantumBackend, QuantumSimulator
from .circuit import Circuit
from .measurement import (
    ExpectationValues,
    expectation_values_to_real,
    concatenate_expectation_values,
)
from openfermion import SymbolicOperator, QubitOperator, IsingOperator
from overrides import overrides
import numpy as np
import pyquil
from typing import Tuple, Optional


def get_context_selection_circuit(
    term: Tuple[Tuple[int, str], ...]
) -> Tuple[Circuit, IsingOperator]:
    """Get the context selection circuit for measuring the expectation value
    of a Pauli term.

    Args:
        term: The Pauli term, expressed using the OpenFermion convention.
    
    Returns:
        Tuple containing:
        - The context selection circuit.
        - The frame operator
    """

    context_selection_circuit = Circuit()
    operator = IsingOperator(())
    for factor in term:
        if factor[1] == "X":
            context_selection_circuit += Circuit(pyquil.gates.RY(-np.pi / 2, factor[0]))
        elif factor[1] == "Y":
            context_selection_circuit += Circuit(pyquil.gates.RX(np.pi / 2, factor[0]))
        operator *= IsingOperator((factor[0], "Z"))

    return context_selection_circuit, operator


class BasicEstimator(Estimator):
    """An estimator that uses the standard approach to computing expectation values of an operator.
    """

    @overrides
    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: Optional[int] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> ExpectationValues:
        """Given a circuit, backend, and target operators, this method produces expectation values 
        for each target operator using the get_expectation_values method built into the provided QuantumBackend. 

        Args:
            backend (QuantumBackend): the backend that will be used to run the circuit
            circuit (Circuit): the circuit that prepares the state.
            target_operator (List[SymbolicOperator]): List of target functions to be estimated.
            n_samples (int): Number of measurements done. 
            epsilon (float): an error term.
            delta (float): a confidence term.

        Returns:
            ExpectationValues: expectation values for each term in the target operator.
        """
        estimator_name = type(self).__name__
        self._log_ignore_parameter(estimator_name, "epsilon", epsilon)
        self._log_ignore_parameter(estimator_name, "delta", delta)

        frame_operators = []
        frame_circuits = []
        for term in target_operator.terms:
            frame_circuit, frame_operator = get_context_selection_circuit(term)
            frame_circuits.append(circuit + frame_circuit)
            frame_operators.append(target_operator.terms[term] * frame_operator)

        if n_samples is not None:
            self.logger.warning(
                "Using n_samples={} (argument passed to get_estimated_expectation_values). Ignoring backend.n_samples={}.".format(
                    n_samples, backend.n_samples
                )
            )
            saved_n_samples = backend.n_samples
            backend.n_samples = n_samples
            measurements_set = backend.run_circuitset_and_measure(frame_circuits)
            backend.n_samples = saved_n_samples
        else:
            measurements_set = backend.run_circuitset_and_measure(frame_circuits)

        expectation_values_set = []
        for frame_operator, measurements in zip(frame_operators, measurements_set):
            expectation_values_set.append(
                expectation_values_to_real(
                    measurements.get_expectation_values(frame_operator)
                )
            )

        return expectation_values_to_real(
            concatenate_expectation_values(expectation_values_set)
        )


class ExactEstimator(Estimator):
    """An estimator that exactly computes the expectation values of an operator. This estimator must run on a quantum simulator. 
    """

    @overrides
    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: Optional[int] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> ExpectationValues:
        """Given a circuit, backend, and target operators, this method produces expectation values 
        for each target operator using the get_exact_expectation_values method built into the provided QuantumBackend. 

        Args:
            backend (QuantumBackend): the backend that will be used to run the circuit
            circuit (Circuit): the circuit that prepares the state.
            target_operator (List[SymbolicOperator]): List of target functions to be estimated.
            n_samples (int): Number of measurements done on the unknown quantum state. 
            epsilon (float): an error term.
            delta (float): a confidence term.

        Raises:
            AttributeError: If backend is not a QuantumSimulator. 

        Returns:
            ExpectationValues: expectation values for each term in the target operator.
        """
        estimator_name = type(self).__name__
        self._log_ignore_parameter(estimator_name, "n_samples", n_samples)
        self._log_ignore_parameter(estimator_name, "epsilon", epsilon)
        self._log_ignore_parameter(estimator_name, "delta", delta)

        if isinstance(backend, QuantumSimulator):
            return backend.get_exact_expectation_values(circuit, target_operator)
        else:
            raise AttributeError(
                "To use the ExactEstimator, the backend must be a QuantumSimulator."
            )


class NoisyEstimator(Estimator):
    """An estimator that simulates the expectation values exactly and incorporate noise in each expectation
        according to a budget of measurements assigned based on a weighted hybrid sampling approach or
        according to a final precision (epsilon) for the expectation value of the target operator, where
        the precision per term is also assignsed based on the weighted hybrid approach.
        https://arxiv.org/abs/2004.06252 
    """

    @overrides
    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: Optional[int] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> ExpectationValues:
        """Given a circuit, backend, and target operators, this method produces expectation values 
        for each target operator using the get_exact_expectation_values method built into the provided QuantumBackend,
        then it simulates the effect of shot noise by adding noise according to the total number of measurements assigned 
        or the precision required.

        Args:
            backend (QuantumBackend): the backend that will be used to run the circuit
            circuit (Circuit): the circuit that prepares the state.
            target_operator (List[SymbolicOperator]): List of target functions to be estimated.
            n_samples (int): Number of measurements done on the unknown quantum state. 
            epsilon (float): an error term.
            delta (float): a confidence term.

        Raises:
            AttributeError: If backend is not a QuantumSimulator. 

        Returns:
            ExpectationValues: expectation values for each term in the target operator.
        """
        estimator_name = type(self).__name__
        self._log_ignore_parameter(estimator_name, "delta", delta)

        if n_samples is not None and epsilon is not None:
            raise AssertionError(
                "For the noisy estimator, either n_samples or epsilon must be provided."
            )

        if isinstance(backend, QuantumSimulator):
            exact_expectations = backend.get_exact_expectation_values(
                circuit, target_operator
            ).values
        else:
            raise AttributeError(
                "To use the NoisyEstimator, the backend must be a QuantumSimulator."
            )

        # compute weights
        weights = []
        constant_position = None
        for position, term in enumerate(target_operator.terms):
            weights.append(abs(target_operator.terms[term]))
            if term == ():
                constant_position = position
        weights = np.array(weights)

        # remove weights and expectation for constant term
        if constant_position is not None:
            weights = np.delete(weights, constant_position)
            exact_expectations = np.delete(exact_expectations, constant_position)

        # compute minimal number of measurements for deterministic strategy
        total_weight = np.sum(weights)
        n_samples_floor = np.ceil(total_weight / np.min(weights))
        print(
            "Minimal number of samples required for determistic strategy: {}".format(
                n_samples_floor
            )
        )

        noisy_expectations = np.zeros_like(exact_expectations)

        if n_samples is not None and epsilon is None:

            self._log_ignore_parameter(estimator_name, "epsilon", epsilon)
            # assign measurements by deterministic strategy
            if n_samples >= n_samples_floor:
                measurements_per_term = np.floor(n_samples * weights / total_weight)
                remaining_samples = n_samples - np.sum(measurements_per_term)
            else:
                measurements_per_term = np.zeros(len(weights))
                remaining_samples = n_samples

            # assign measurements by sampling
            if remaining_samples > 0:
                measurements_per_term += np.random.multinomial(
                    remaining_samples, weights / total_weight
                )

            # compute noisy expectations
            for index, (expectation, n_measurements) in enumerate(
                zip(exact_expectations, measurements_per_term)
            ):
                probability = (expectation + 1.0) / 2.0
                if probability == 1.0 or probability == 0.0:
                    probability_variance = 1.0 / ((n_measurements + 2) ** 2.0)
                else:
                    probability_variance = (
                        (1.0 - probability) * probability / n_measurements
                    )
                # sampling from a beta distribution
                a, b = self._get_beta_parameters_from_mu_and_sigma(
                    probability, np.sqrt(probability_variance)
                )
                if a <=0 or b<=0:
                    noisy_expectations[index] = 0.0
                else:
                    noisy_expectations[index] = 2.0 * np.random.beta(a, b) - 1.0

        elif n_samples is None and epsilon is not None:

            self._log_ignore_parameter(estimator_name, "n_samples", n_samples)
            # compute required precision per term based on deterministic weighting rule
            precisions_per_term = np.sqrt((epsilon ** 2.0) * weights / total_weight)
            for index, (expectation, precision) in enumerate(
                zip(exact_expectations, precisions_per_term)
            ):
                probability = (expectation + 1.0) / 2.0
                # the factor of 2 comes from the conversion from expectations to probabilities
                a, b = self._get_beta_parameters_from_mu_and_sigma(
                    probability, precision / 2.0
                )
                # sampling from a beta distribution
                if a <=0 or b<=0:
                    noisy_expectations[index] = 0.0
                else:
                    noisy_expectations[index] = 2.0 * np.random.beta(a, b) - 1.0

        # reinserting constant term expectation
        if constant_position is not None:
            noisy_expectations = np.insert(noisy_expectations, constant_position, 1.0)
        return ExpectationValues(noisy_expectations)

    def _get_beta_parameters_from_mu_and_sigma(self, mu: float, sigma: float):
        """
            Auxiliary function to estimate parameters of a beta distribution from
            values of mean (mu) and precision (sigma)
            mu \in (0, 1), sigma \in (0, 0.5)
        """
        alpha = (mu ** 2.0) * ((1.0 - mu) / (sigma ** 2.0) - (1.0 / mu))
        beta = alpha * ((1.0 / mu) - 1.0)
        return alpha, beta
