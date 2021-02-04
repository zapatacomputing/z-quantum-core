from .interfaces.estimator import Estimator
from .interfaces.backend import QuantumBackend, QuantumSimulator
from .circuit import Circuit
from .measurement import (
    ExpectationValues,
    expectation_values_to_real,
    concatenate_expectation_values,
)
from .hamiltonian import get_decomposition_function, estimate_nmeas_for_frames
from .utils import scale_and_discretize
from openfermion import SymbolicOperator, IsingOperator, QubitOperator
from overrides import overrides
import logging
import numpy as np
import pyquil
from typing import Tuple, Optional, Callable, List

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
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
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
            epsilon: Inherited from Estimator, not used.
            delta: Inherited from Estimator, not used.
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
        if shot_allocation_strategy not in ("optimal", "uniform",):
            raise ValueError(
                f"Invalid shot allocation stratgey: ${shot_allocation_strategy}"
            )

        frame_operators = []
        frame_circuits = []
        groups = get_decomposition_function(self.decomposition_method)(target_operator)
        for group in groups:
            frame_circuit, frame_operator = get_context_selection_circuit_for_group(
                group
            )
            frame_circuits.append(circuit + frame_circuit)
            frame_operators.append(frame_operator)

        if shot_allocation_strategy == "uniform":

            if n_total_samples is not None:
                raise ValueError(
                    "Uniform sampling does not yet support n_total_samples."
                )

            if n_samples is not None:
                logger.warning(
                    f"""Using n_samples={n_samples} (argument passed to get_estimated_expectation_values). 
                        Ignoring backend.n_samples={backend.n_samples}"""
                )
                n_samples = (n_samples,) * len(frame_circuits)
                measurements_set = backend.run_circuitset_and_measure(
                    frame_circuits, n_samples
                )
            else:
                measurements_set = backend.run_circuitset_and_measure(frame_circuits)

        elif shot_allocation_strategy == "optimal":
            if n_total_samples is None:
                raise ValueError(
                    "For optimal shot allocation, n_total_samples must be provided."
                )

            if n_samples is not None:
                raise ValueError(
                    "Optimal shot allocation does not support n_samples; use n_total_samples instead."
                )

            K2, nterms, measurements_per_frame = estimate_nmeas_for_frames(
                frame_operators, self.prior_expectation_values
            )

            measurements_per_frame = scale_and_discretize(
                measurements_per_frame, n_total_samples
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
        if isinstance(backend, QuantumSimulator):
            return backend.get_exact_expectation_values(circuit, target_operator)
        else:
            raise AttributeError(
                "To use the ExactEstimator, the backend must be a QuantumSimulator."
            )
