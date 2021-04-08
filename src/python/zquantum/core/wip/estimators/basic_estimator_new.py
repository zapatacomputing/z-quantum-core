from pyquil.gates import S
from .new_estimator import EstimationProblem, EstimateExpectationValues
from ...interfaces.backend import QuantumBackend, QuantumSimulator
from ...measurement import (
    ExpectationValues,
    expectation_values_to_real,
    concatenate_expectation_values,
)
from overrides import overrides
import logging
import numpy as np
from typing import Dict, Tuple, Optional, Callable, List


def naively_estimate_expectation_values(
    backend: QuantumBackend,
    estimation_problems: List[EstimationProblem],
) -> ExpectationValues:

    circuits, operators, shots_per_circuit = zip(
        *[(e.circuit, e.operator, e.number_of_shots) for e in estimation_problems]
    )

    circuits = [
        estimation_problem.circuit for estimation_problem in estimation_problems
    ]
    operators = [
        estimation_problem.operator for estimation_problem in estimation_problems
    ]
    shots_per_circuit = [
        estimation_problem.number_of_shots for estimation_problem in estimation_problems
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
    backend: QuantumBackend,
    estimation_problems: List[EstimationProblem],
) -> ExpectationValues:
    expectation_values_list = [
        backend.get_exact_expectation_values(
            estimation_problem.circuit, estimation_problem.target_operator
        )
        for estimation_problem in estimation_problems
    ]
    return expectation_values_to_real(
        concatenate_expectation_values(expectation_values_list)
    )
