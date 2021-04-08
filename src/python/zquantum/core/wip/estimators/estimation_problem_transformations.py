from typing import List, Tuple, Optional
from openfermion import QubitOperator, IsingOperator
import pyquil
import numpy as np

# from ..circuit import Circuit, Gate
# TODO: remove this
from ...circuit._circuit import Circuit
from ...circuit._gate import Gate
from .new_estimator import (
    EstimationProblem,
    EstimationProblemTransformer,
)
from ...hamiltonian import group_comeasureable_terms_greedy, estimate_nmeas_for_frames
from ...utils import scale_and_discretize
from ...measurement import ExpectationValues


greedy_grouping_with_context_selection: EstimationProblemTransformer
uniform_shot_allocation: EstimationProblemTransformer
optimal_shot_allocation: EstimationProblemTransformer


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
            if (qubit, operator) not in context:
                context.append((qubit, operator))
            term_operator *= IsingOperator((qubit, "Z"))
        transformed_operator += term_operator * qubit_operator.terms[term]

    for factor in context:
        if factor[1] == "X":
            context_selection_circuit += Circuit(pyquil.gates.RY(-np.pi / 2, factor[0]))
        elif factor[1] == "Y":
            context_selection_circuit += Circuit(pyquil.gates.RX(np.pi / 2, factor[0]))

    return context_selection_circuit, transformed_operator


def greedy_grouping_with_context_selection(
    estimation_problems: List[EstimationProblem],
) -> List[EstimationProblem]:
    """
    TODO
    """
    output_estimation_problems = []
    for estimation_problem in estimation_problems:
        groups = group_comeasureable_terms_greedy(estimation_problem.operator)
        for group in groups:
            (
                context_selection_circuit,
                frame_operator,
            ) = get_context_selection_circuit_for_group(group)
            frame_circuit = estimation_problem.circuit + context_selection_circuit
            group_estimation_problem = EstimationProblem(
                frame_operator, frame_circuit, estimation_problem.constraints
            )
            output_estimation_problems.append(group_estimation_problem)
    return output_estimation_problems


def uniform_shot_allocation(number_of_shots: int) -> EstimationProblemTransformer:
    """
    TODO
    """

    def _allocate_shots(
        estimation_problems: List[EstimationProblem],
    ) -> List[EstimationProblem]:

        return [
            EstimationProblem(
                operator=estimation_problem.operator,
                circuit=estimation_problem.circuit,
                number_of_shots=number_of_shots,
            )
            for estimation_problem in estimation_problems
        ]

    return _allocate_shots


def optimal_shot_allocation(
    total_n_shots: int,
    prior_expectation_values: Optional[ExpectationValues] = None,
) -> EstimationProblemTransformer:
    """
    TODO
    """

    def _allocate_shots(
        estimation_problems: List[EstimationProblem],
    ) -> List[EstimationProblem]:
        frame_operators = [
            estimation_problem.operator for estimation_problem in estimation_problems
        ]

        _, _, measurements_per_frame = estimate_nmeas_for_frames(
            frame_operators, prior_expectation_values
        )

        measurements_per_frame = scale_and_discretize(
            measurements_per_frame, total_n_shots
        )

        return [
            EstimationProblem(
                operator=estimation_problem.operator,
                circuit=estimation_problem.circuit,
                number_of_shots=number_of_shots,
            )
            for estimation_problem, number_of_shots in zip(
                estimation_problems, measurements_per_frame
            )
        ]

    return _allocate_shots
