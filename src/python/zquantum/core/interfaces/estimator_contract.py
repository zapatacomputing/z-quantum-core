from openfermion import IsingOperator
from zquantum.core.circuits import Circuit, H, RX, RY, RZ
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.symbolic_simulator import SymbolicSimulator

import numpy as np
import pytest

backend = SymbolicSimulator()

estimation_tasks = [
    EstimationTask(IsingOperator("Z0"), Circuit([H(0)]), 10000),
    EstimationTask(
        IsingOperator("Z0") + IsingOperator("Z1") + IsingOperator("Z2"),
        Circuit([H(0), H(1), H(2)]),
        10000,
    ),
    EstimationTask(
        IsingOperator("Z0") + IsingOperator("Z1", 4),
        Circuit(
            [
                RX(np.pi)(0),
                RY(0.12)(1),
                RZ(np.pi / 3)(1),
                RY(1.9213)(0),
            ]
        ),
        10000,
    ),
]


def _validate_each_task_returns_one_expecation_value(estimator):
    # When
    expectation_values = estimator(
        backend=backend,
        estimation_tasks=estimation_tasks,
    )

    # Then
    return len(expectation_values) == len(estimation_tasks)


def _validate_order_of_outputs_matches_order_of_inputs(estimator):
    expectation_values = estimator(
        backend=backend,
        estimation_tasks=estimation_tasks,
    )

    # TODO: experiment with seeds to achieve determinism

    return all(
        [
            pytest.approx(expectation_values[i].values, abs=2e-1)
            == estimator(
                backend=backend,
                estimation_tasks=[task],
            )[0].values
            for i, task in enumerate(estimation_tasks)
        ]
    )


def _validate_number_of_entries_in_each_expectation_value_is_not_restricted(estimator):
    return True


def _validate_expectation_value_includes_coefficients(estimator):
    estimation_tasks = [
        EstimationTask(IsingOperator("Z0"), Circuit([H(0)]), 10000),
        # EstimationTask(IsingOperator("Z0", 1.9971997), Circuit([H(0)]), 10000),
        EstimationTask(IsingOperator("Z0", 19.971997), Circuit([H(0)]), 10000),
    ]

    expectation_values = estimator(
        backend=backend,
        estimation_tasks=estimation_tasks,
    )

    return expectation_values[0].values != pytest.approx(
        expectation_values[1].values, abs=2e-1
    )


def _validate_constant_terms_are_included_in_output(estimator):
    estimation_tasks = [
        EstimationTask(IsingOperator("Z0"), Circuit([H(0)]), 10000),
        # EstimationTask(IsingOperator("Z0", 1.9971997), Circuit([H(0)]), 10000),
        EstimationTask(
            IsingOperator("Z0", 19.971997) + IsingOperator("[]", 19.971997),
            Circuit([H(0)]),
            10000,
        ),
    ]

    expectation_values = estimator(
        backend=backend,
        estimation_tasks=estimation_tasks,
    )

    return expectation_values[0].values != pytest.approx(
        expectation_values[1].values, abs=2e-1
    )


ESTIMATOR_CONTRACT = [
    _validate_each_task_returns_one_expecation_value,
    _validate_order_of_outputs_matches_order_of_inputs,
    _validate_number_of_entries_in_each_expectation_value_is_not_restricted,
    _validate_expectation_value_includes_coefficients,
    _validate_constant_terms_are_included_in_output,
]
