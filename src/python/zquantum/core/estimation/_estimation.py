from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import sympy
from openfermion import IsingOperator, QubitOperator

from ..circuits import RX, RY, Circuit
from ..hamiltonian import estimate_nmeas_for_frames, group_comeasureable_terms_greedy
from ..interfaces.backend import QuantumBackend, QuantumSimulator
from ..interfaces.estimation import EstimationTask
from ..measurement import ExpectationValues, expectation_values_to_real
from ..openfermion import change_operator_type
from ..utils import scale_and_discretize


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
    context: List[Tuple[int, str]] = []

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
            context_selection_circuit += RY(-np.pi / 2)(factor[0])
        elif factor[1] == "Y":
            context_selection_circuit += RX(np.pi / 2)(factor[0])

    return context_selection_circuit, transformed_operator


def perform_context_selection(
    estimation_tasks: List[EstimationTask],
) -> List[EstimationTask]:
    """Changes the circuits in estimation tasks to involve context selection.

    Args:
        estimation_tasks: list of estimation tasks
    """
    output_estimation_tasks = []
    for estimation_task in estimation_tasks:
        (
            context_selection_circuit,
            frame_operator,
        ) = get_context_selection_circuit_for_group(estimation_task.operator)
        frame_circuit = estimation_task.circuit + context_selection_circuit
        new_estimation_task = EstimationTask(
            frame_operator, frame_circuit, estimation_task.number_of_shots
        )
        output_estimation_tasks.append(new_estimation_task)
    return output_estimation_tasks


def group_individually(estimation_tasks: List[EstimationTask]) -> List[EstimationTask]:
    """
    Transforms list of estimation tasks by putting each term into a estimation task.

    Args:
        estimation_tasks: list of estimation tasks

    """
    output_estimation_tasks = []
    for estimation_task in estimation_tasks:
        for term in estimation_task.operator.get_operators():
            output_estimation_tasks.append(
                EstimationTask(
                    term, estimation_task.circuit, estimation_task.number_of_shots
                )
            )
    return output_estimation_tasks


def group_greedily(
    estimation_tasks: List[EstimationTask], sort_terms: bool = False
) -> List[EstimationTask]:
    """
    Transforms list of estimation tasks by performing greedy grouping and adding
    context selection logic to the circuits.

    Args:
        estimation_tasks: list of estimation tasks
    """
    if sort_terms:
        print("Greedy grouping with pre-sorting")
    else:
        print("Greedy grouping without pre-sorting")
    output_estimation_tasks = []
    for estimation_task in estimation_tasks:
        groups = group_comeasureable_terms_greedy(
            estimation_task.operator, sort_terms=sort_terms
        )
        for group in groups:
            group_estimation_task = EstimationTask(
                group, estimation_task.circuit, estimation_task.number_of_shots
            )
            output_estimation_tasks.append(group_estimation_task)
    return output_estimation_tasks


def allocate_shots_uniformly(
    estimation_tasks: List[EstimationTask], number_of_shots: int
) -> List[EstimationTask]:
    """
    Allocates the same number of shots to each task.

    Args:
        number_of_shots: number of shots to be assigned to each EstimationTask
    """
    if number_of_shots <= 0:
        raise ValueError("number_of_shots must be positive.")

    return [
        EstimationTask(
            operator=estimation_task.operator,
            circuit=estimation_task.circuit,
            number_of_shots=number_of_shots,
        )
        for estimation_task in estimation_tasks
    ]


def allocate_shots_proportionally(
    estimation_tasks: List[EstimationTask],
    total_n_shots: int,
    prior_expectation_values: Optional[ExpectationValues] = None,
) -> List[EstimationTask]:
    """Allocates specified number of shots proportionally to the variance associated
    with each operator in a list of estimation tasks. For more details please refer to
    the documentation of `zquantum.core.hamiltonian.estimate_nmeas_for_frames`.

    Args:
        total_n_shots: total number of shots to be allocated
        prior_expectation_values: object containing the expectation
            values of all operators in frame_operators
    """
    if total_n_shots <= 0:
        raise ValueError("total_n_shots must be positive.")

    frame_operators = [estimation_task.operator for estimation_task in estimation_tasks]

    _, _, relative_measurements_per_frame = estimate_nmeas_for_frames(
        frame_operators, prior_expectation_values
    )

    measurements_per_frame = scale_and_discretize(
        relative_measurements_per_frame, total_n_shots
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


def evaluate_estimation_circuits(
    estimation_tasks: List[EstimationTask],
    symbols_maps: List[Dict[sympy.Symbol, float]],
) -> List[EstimationTask]:
    """Evaluates circuits given in all estimation tasks using the given symbols_maps.

    If one symbols map is given, it is used to evaluate all circuits. Otherwise, the
    symbols map at index i will be used for the estimation task at index i.

    Args:
        estimation_tasks: the estimation tasks which contain the circuits to be
            evaluated
        symbols_maps: a list of dictionaries (or singular dictionary) that map the
            symbolic symbols used in the parametrized circuits to the associated values
    """
    return [
        EstimationTask(
            operator=estimation_task.operator,
            circuit=estimation_task.circuit.bind(symbols_map),
            number_of_shots=estimation_task.number_of_shots,
        )
        for estimation_task, symbols_map in zip(estimation_tasks, symbols_maps)
    ]


def split_constant_estimation_tasks(
    estimation_tasks: List[EstimationTask],
) -> Tuple[List[EstimationTask], List[EstimationTask], List[int], List[int]]:
    """This function splits a given list of EstimationTask into two: one that
    contains EstimationTasks that only contain constants, and one that contains
    EstimationTasks that have non-constant terms as well.
    that require 0 shot generate an error.

    Args:
        estimation_tasks: The list of estimation tasks for which
                         Expectation Values are wanted.

    Returns:
        estimation_tasks_to_measure: A new list of estimation tasks that only
            contains the ones that should actually be submitted to the backend
        estimation_tasks_for_constants: A new list of estimation tasks that
            contains the EstimationTasks with only constant terms
        indices_to_measure: A list containing the indices of the EstimationTasks we will
            actually measure, i.e. the ith estimation_tasks_to_measure expectation
            value will go into the indices_to_measure[i] position.
        indices_for_constants: A list containing the indices of the EstimationTasks for
            constant terms.
    """

    estimation_tasks_to_measure = []
    estimation_tasks_for_constants = []
    indices_to_measure = []
    indices_for_constants = []
    for i, task in enumerate(estimation_tasks):
        if len(task.operator.terms) == 1 and () in task.operator.terms.keys():
            indices_for_constants.append(i)
            estimation_tasks_for_constants.append(task)
        elif task.number_of_shots == 0:
            raise RuntimeError(
                "An EstimationTask requested 0 shot for a non-constant term. "
                "It's unclear what to do with that."
            )
        else:
            indices_to_measure.append(i)
            estimation_tasks_to_measure.append(task)

    return (
        estimation_tasks_to_measure,
        estimation_tasks_for_constants,
        indices_to_measure,
        indices_for_constants,
    )


def evaluate_constant_estimation_tasks(
    estimation_tasks: List[EstimationTask],
) -> List[ExpectationValues]:
    """This function evaluates a list of EstimationTask over constant terms.

    Args:
        estimation_tasks: The list of estimation tasks for which
            Expectation Values are wanted, they must only contain constant terms.

    Returns:
        expectation_values: the expectation values over constant terms,
            with their correlations and estimator_covariances.
    """

    expectation_values = []
    for task in estimation_tasks:
        if len(task.operator.terms) > 1 or () not in task.operator.terms.keys():
            raise RuntimeError(
                "evaluate_constant_estimation_tasks received an EstimationTask "
                "that contained a non-constant term."
            )
        coefficient = task.operator.terms[()]
        expectation_values.append(
            ExpectationValues(
                np.asarray([coefficient]),
                correlations=[np.asarray([[0.0]])],
                estimator_covariances=[np.asarray([[0.0]])],
            )
        )

    return expectation_values


def estimate_expectation_values_by_averaging(
    backend: QuantumBackend,
    estimation_tasks: List[EstimationTask],
) -> List[ExpectationValues]:
    """Basic method for estimating expectation values for list of estimation tasks.

    It executes specified circuit and calculates expectation values based on the
    measurements.

    Args:
        backend: backend used for executing circuits
        estimation_tasks: list of estimation tasks
    """

    (
        estimation_tasks_to_measure,
        estimation_tasks_for_constants,
        indices_to_measure,
        indices_for_constants,
    ) = split_constant_estimation_tasks(estimation_tasks)

    expectation_values_for_constants = evaluate_constant_estimation_tasks(
        estimation_tasks_for_constants
    )

    circuits, operators, shots_per_circuit = zip(
        *[
            (e.circuit, e.operator, e.number_of_shots)
            for e in estimation_tasks_to_measure
        ]
    )

    measurements_list = backend.run_circuitset_and_measure(circuits, shots_per_circuit)

    measured_expectation_values_list = [
        expectation_values_to_real(
            measurements.get_expectation_values(
                change_operator_type(frame_operator, IsingOperator)
            )
        )
        for frame_operator, measurements in zip(operators, measurements_list)
    ]

    full_expectation_values: List[Optional[ExpectationValues]] = [
        None
        for _ in range(
            len(estimation_tasks_for_constants) + len(estimation_tasks_to_measure)
        )
    ]

    for ex_val, final_index in zip(
        expectation_values_for_constants, indices_for_constants
    ):
        full_expectation_values[final_index] = ex_val
    for ex_val, final_index in zip(
        measured_expectation_values_list, indices_to_measure
    ):
        full_expectation_values[final_index] = ex_val

    return cast(List[ExpectationValues], full_expectation_values)


def calculate_exact_expectation_values(
    backend: QuantumSimulator,
    estimation_tasks: List[EstimationTask],
) -> List[ExpectationValues]:
    """Calculates exact expectation values using built-in method of a provided backend.

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
    return expectation_values_list
