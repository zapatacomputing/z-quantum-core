from pyquil.gates import S
from .estimator_interface import EstimationTask
from ...interfaces.backend import QuantumBackend, QuantumSimulator
from ...measurement import (
    ExpectationValues,
    expectation_values_to_real,
    concatenate_expectation_values,
)
from typing import List


def naively_estimate_expectation_values(
    backend: QuantumBackend,
    estimation_tasks: List[EstimationTask],
) -> ExpectationValues:

    circuits, operators, shots_per_circuit = zip(
        *[(e.circuit, e.operator, e.number_of_shots) for e in estimation_tasks]
    )

    circuits = [estimation_task.circuit for estimation_task in estimation_tasks]
    operators = [estimation_task.operator for estimation_task in estimation_tasks]
    shots_per_circuit = [
        estimation_task.number_of_shots for estimation_task in estimation_tasks
    ]
    measurements_list = backend.run_circuitset_and_measure(circuits, shots_per_circuit)

    expectation_values_list = [
        expectation_values_to_real(measurements.get_expectation_values(frame_operator))
        for frame_operator, measurements in zip(operators, measurements_list)
    ]

    # TODO handle empty term?
    # if target_operator.terms.get(()) is not None:
    #     expectation_values_set.append(
    #         ExpectationValues(np.array([target_operator.terms.get(())]))
    #     )

    return expectation_values_to_real(
        concatenate_expectation_values(expectation_values_list)
    )


def calculate_exact_expectation_values(
    backend: QuantumSimulator,
    estimation_tasks: List[EstimationTask],
) -> ExpectationValues:
    expectation_values_list = [
        backend.get_exact_expectation_values(
            estimation_task.circuit, estimation_task.target_operator
        )
        for estimation_task in estimation_tasks
    ]
    return expectation_values_to_real(
        concatenate_expectation_values(expectation_values_list)
    )
