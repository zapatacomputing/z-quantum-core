import logging
from typing import List, Optional, Tuple

import numpy as np
import pyquil
from openfermion import IsingOperator, QubitOperator, SymbolicOperator
from overrides import overrides

from .circuit import Circuit
from .hamiltonian import estimate_nmeas_for_frames, get_decomposition_function
from .interfaces.backend import QuantumBackend, QuantumSimulator
from .interfaces.estimator import Estimator
from .measurement import (
    ExpectationValues,
    concatenate_expectation_values,
    expectation_values_to_real,
)
from .utils import scale_and_discretize

logger = logging.getLogger(__name__)


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


def get_context_selection_circuit_for_group(
    qubit_operator: QubitOperator,
) -> Tuple[Circuit, IsingOperator]:
    """Get the context selection circuit for measuring the expectation value
    of a group of co-measurable Pauli terms.

    Args:
        term: The Pauli term, expressed using the OpenFermion convention.

    Returns:
        Tuple containing:
        - The context selection circuit.
        - The frame operator
    """

    context_selection_circuit = Circuit()
    transformed_operator = IsingOperator()
    context = []

    for term in qubit_operator.terms:
        term_operator = IsingOperator(())
        for qubit, operator in term:
            for existing_qubit, existing_operator in context:
                if existing_qubit == qubit and existing_operator != operator:
                    raise ValueError("Terms are not co-measurable")
            if not (qubit, operator) in context:
                context.append((qubit, operator))
            term_operator *= IsingOperator((qubit, "Z"))
        transformed_operator += term_operator * qubit_operator.terms[term]

    for factor in context:
        if factor[1] == "X":
            context_selection_circuit += Circuit(pyquil.gates.RY(-np.pi / 2, factor[0]))
        elif factor[1] == "Y":
            context_selection_circuit += Circuit(pyquil.gates.RX(np.pi / 2, factor[0]))

    return context_selection_circuit, transformed_operator


def allocate_shots(
    shot_allocation_strategy: str,
    frame_operators: List[IsingOperator],
    n_samples: Optional[int] = None,
    n_total_samples: Optional[int] = None,
    prior_expectation_values: Optional[ExpectationValues] = None,
) -> List[int]:
    """Generates the number of shots for each frame operator, using either the "uniform"
    or the "optimal" shot allocation.

    Args:
        shot_allocation_strategy: Which shot allocation strategy to use. "uniform" attributes the same
                                  number of shots to each frame operator, whereas "optimal" divides the
                                  shots among all frame_operators according to their coefficients and
                                  variances if prior expectation values are provided.
        frame_operators: The list of IsingOperators that will be evaluated
        n_samples: The number of samples per frame operator using the uniform allocation strategy
                   Exactly one of n_samples or n_total_samples must be provided
        n_total_samples: The total number of samples across all frame operators when using the optimal
                         allocation strategy.
                         Exactly one of n_samples or n_total_samples must be provided
        prior_expectation_values: If given, this is used to estimate the variances of the frame operators when
                                  doing optimal shot allocation.
    Returns:
        List of integers giving the number of shots for each frame operator.
    """

    if shot_allocation_strategy == "uniform":
        if n_total_samples is not None:
            raise ValueError("Uniform sampling does not yet support n_total_samples.")
        if n_samples is not None:
            measurements_per_frame = [n_samples for _ in range(len(frame_operators))]
        else:
            measurements_per_frame = None

    elif shot_allocation_strategy == "optimal":
        if n_total_samples is None:
            raise ValueError(
                "For optimal shot allocation, n_total_samples must be provided."
            )
        if n_samples is not None:
            raise ValueError(
                "Optimal shot allocation does not support n_samples; use n_total_samples instead."
            )
        _, _, measurements_per_frame = estimate_nmeas_for_frames(
            frame_operators, prior_expectation_values
        )
        measurements_per_frame = scale_and_discretize(
            measurements_per_frame, n_total_samples
        )
    else:
        raise ValueError(
            f"Invalid shot allocation stratgey: ${shot_allocation_strategy}"
        )

    return measurements_per_frame


class BasicEstimator(Estimator):
    """An estimator that uses the standard approach to computing expectation values of an operator.

    Attributes:
        decomposition_method (str): Which Hamiltonian decomposition method
            to use. Available options are: 'greedy-sorted' (default) and
            'greedy'.
    """

    def __init__(
        self, decomposition_method: str = "greedy-sorted", prior_expectation_values=None
    ):
        self.decomposition_method = decomposition_method
        self.prior_expectation_values = prior_expectation_values

    @overrides
    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: Optional[int] = None,
        n_total_samples: Optional[int] = None,
        shot_allocation_strategy: str = "uniform",
    ) -> ExpectationValues:
        """Given a circuit, backend, and target operators, this method produces expectation values
        for each target operator using the get_expectation_values method built into the provided QuantumBackend.

        Args:
            backend: the backend that will be used to run the circuit
            circuit: the circuit that prepares the state.
            target_operator): List of target functions to be estimated.
            n_samples: Number of measurements to be performed on each frame.
                Exactly one of n_samples and n_total_samples must be provided.
            n_total_samples: Total number of measurements to be performed across
                all frames. Exactly one of n_samples and n_total_samples must be
                provided.
            shot_allocation_strategy: Strategy for allocating shots to groups.
                - "uniform": The number of shots specified by n_samples is used
                    for each group.
                - "optimal": The number of shots specified by n_samples is
                    divided amongst the groups optimally. If
                    self.prior_expectation_values is set, it will be used to
                    account for variances. Otherwise the upper bound on
                    variances is used. Covariances are assumed to be zero.

        Returns:
            ExpectationValues: expectation values for each term in the target operator.
        """
        frame_operators = []
        frame_circuits = []
        groups = get_decomposition_function(self.decomposition_method)(target_operator)
        for group in groups:
            frame_circuit, frame_operator = get_context_selection_circuit_for_group(
                group
            )
            frame_circuits.append(circuit + frame_circuit)
            frame_operators.append(frame_operator)

        measurements_per_frame = allocate_shots(
            shot_allocation_strategy,
            frame_operators,
            n_samples,
            n_total_samples,
            self.prior_expectation_values,
        )

        measurements_set = backend.run_circuitset_and_measure(
            frame_circuits, measurements_per_frame
        )

        expectation_values_set = []
        for frame_operator, measurements in zip(frame_operators, measurements_set):
            expectation_values_set.append(
                expectation_values_to_real(
                    measurements.get_expectation_values(frame_operator)
                )
            )

        if target_operator.terms.get(()) is not None:
            expectation_values_set.append(
                ExpectationValues(np.array([target_operator.terms.get(())]))
            )

        return expectation_values_to_real(
            concatenate_expectation_values(expectation_values_set)
        )


class ExactEstimator(Estimator):
    """An estimator that exactly computes the expectation values of an operator. This estimator must run on a quantum simulator."""

    @overrides
    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: Optional[int] = None,
    ) -> ExpectationValues:
        """Given a circuit, backend, and target operators, this method produces expectation values
        for each target operator using the get_exact_expectation_values method built into the provided QuantumBackend.

        Args:
            backend (QuantumBackend): the backend that will be used to run the circuit
            circuit (Circuit): the circuit that prepares the state.
            target_operator (List[SymbolicOperator]): List of target functions to be estimated.
            n_samples (int): Number of measurements done on the unknown quantum state.

        Raises:
            AttributeError: If backend is not a QuantumSimulator.

        Returns:
            ExpectationValues: expectation values for each term in the target operator.
        """
        if isinstance(backend, QuantumSimulator):
            return backend.get_exact_expectation_values(circuit, target_operator)
        else:
            raise AttributeError(
                "To use the ExactEstimator, the backend must be a QuantumSimulator."
            )
