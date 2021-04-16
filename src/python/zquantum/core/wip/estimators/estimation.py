from typing import List, Optional, Tuple

import numpy as np
import pyquil
from openfermion import IsingOperator, QubitOperator

from ...circuit._circuit import Circuit
from ...hamiltonian import estimate_nmeas_for_frames, group_comeasureable_terms_greedy
from ...interfaces.backend import QuantumBackend, QuantumSimulator
from ...measurement import (
    ExpectationValues,
    concatenate_expectation_values,
    expectation_values_to_real,
)
from ...utils import scale_and_discretize
from .estimation_interface import EstimationTask, EstimationTaskTransformer


def get_context_selection_circuit(
    operator: QubitOperator,
) -> Tuple[Circuit, IsingOperator]:
    """Get the context selection circuit for measuring the expectation value
    of a Pauli term.

    Args:
        operator: operator consisting of a single Pauli Term

    """
    context_selection_circuit = Circuit()
    output_operator = IsingOperator(())
    terms = list(operator.terms.keys())[0]
    for term in terms:
        term_type = term[1]
        qubit_id = term[0]
        if term_type == "X":
            context_selection_circuit += Circuit(pyquil.gates.RY(-np.pi / 2, qubit_id))
        elif term_type == "Y":
            context_selection_circuit += Circuit(pyquil.gates.RX(np.pi / 2, qubit_id))
        output_operator *= IsingOperator((qubit_id, "Z"))
    return context_selection_circuit, output_operator


def get_context_selection_circuit_for_group(
    qubit_operator: QubitOperator,
) -> Tuple[Circuit, IsingOperator]:
    """Get the context selection circuit for measuring the expectation value
    of a group of co-measurable Pauli terms.

    Args:
        qubit_operator: operator representing group of co-measurable Pauli term
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
    estimation_tasks: List[EstimationTask],
) -> List[EstimationTask]:
    """
    Transforms list of estimation tasks by performing greedy grouping and adding
    context selection logic to the circuits.

    Args:
        estimation_tasks: list of estimation tasks
    """
    output_estimation_tasks = []
    for estimation_task in estimation_tasks:
        groups = group_comeasureable_terms_greedy(estimation_task.operator)
        for group in groups:
            (
                context_selection_circuit,
                frame_operator,
            ) = get_context_selection_circuit_for_group(group)
            frame_circuit = estimation_task.circuit + context_selection_circuit
            group_estimation_task = EstimationTask(
                frame_operator, frame_circuit, estimation_task.constraints
            )
            output_estimation_tasks.append(group_estimation_task)
    return output_estimation_tasks


def uniform_shot_allocation(number_of_shots: int) -> EstimationTaskTransformer:
    """
    Returns an EstimationTaskTransformer which allocates the same number of shots to each task.

    Args:
        number_of_shots: number of shots to be assigned to each EstimationTask
    """
    if number_of_shots <= 0:
        raise ValueError("number_of_shots must be positive.")

    def _allocate_shots(
        estimation_tasks: List[EstimationTask],
    ) -> List[EstimationTask]:

        return [
            EstimationTask(
                operator=estimation_task.operator,
                circuit=estimation_task.circuit,
                number_of_shots=number_of_shots,
            )
            for estimation_task in estimation_tasks
        ]

    return _allocate_shots


def proportional_shot_allocation(
    total_n_shots: int,
    prior_expectation_values: Optional[ExpectationValues] = None,
) -> EstimationTaskTransformer:
    """
    Returns an EstimationTaskTransformer which allocates the same number of shots to each task.
    For more details please refer to documentation of zquantum.core.hamiltonian.estimate_nmeas_for_frames .

    Args:
        total_n_shots: total number of shots to be allocated
        prior_expectation_values: object containing the expectation
            values of all operators in frame_operators
    """
    if total_n_shots <= 0:
        raise ValueError("total_n_shots must be positive.")

    def _allocate_shots(
        estimation_tasks: List[EstimationTask],
    ) -> List[EstimationTask]:
        frame_operators = [
            estimation_task.operator for estimation_task in estimation_tasks
        ]

        _, _, measurements_per_frame = estimate_nmeas_for_frames(
            frame_operators, prior_expectation_values
        )

        measurements_per_frame = scale_and_discretize(
            measurements_per_frame, total_n_shots
        )

        return [
            EstimationTask(
                operator=estimation_task.operator,
                circuit=estimation_task.circuit,
                number_of_shots=number_of_shots,
            )
            for estimation_task, number_of_shots in zip(
                estimation_tasks, measurements_per_frame
            )
        ]

    return _allocate_shots


def naively_estimate_expectation_values(
    backend: QuantumBackend,
    estimation_tasks: List[EstimationTask],
) -> ExpectationValues:
    """
    Basic method for estimating expectation values for list of estimation tasks.
    It executes specified circuit and calculates expectation values based on the measurements.

    Args:
        backend: backend used for executing circuits
        estimation_tasks: list of estimation tasks
    """
    circuits, operators, shots_per_circuit = zip(
        *[(e.circuit, e.operator, e.number_of_shots) for e in estimation_tasks]
    )

    measurements_list = backend.run_circuitset_and_measure(circuits, shots_per_circuit)

    expectation_values_list = [
        expectation_values_to_real(measurements.get_expectation_values(frame_operator))
        for frame_operator, measurements in zip(operators, measurements_list)
    ]

    # TODO handle empty term?
    # if operator.terms.get(()) is not None:
    #     expectation_values_set.append(
    #         ExpectationValues(np.array([operator.terms.get(())]))
    #     )

    return expectation_values_to_real(
        concatenate_expectation_values(expectation_values_list)
    )


def calculate_exact_expectation_values(
    backend: QuantumSimulator,
    estimation_tasks: List[EstimationTask],
) -> ExpectationValues:
    """
    Calculates exact expectation values using built-in method of a provided backend.

    Args:
        backend: backend used for executing circuits
        estimation_tasks: list of estimation tasks
    """
    expectation_values_list = [
        backend.get_exact_expectation_values(
            estimation_task.circuit, estimation_task.operator
        )
        for estimation_task in estimation_tasks
    ]
    return expectation_values_to_real(
        concatenate_expectation_values(expectation_values_list)
    )
