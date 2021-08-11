"""Test case prototypes of instances of the EstimateExpectationValues protocol
that can be used in other projects.

Note that this file won't be executed on its own by pytest.
You need to define your own test cases that import the ones defined here.
Here is an example of how you would do that:

    from zquantum.core.interfaces.estimator_contract import ESTIMATOR_CONTRACTS

    @pytest.mark.parametrize("contract", ESTIMATOR_CONTRACTS)
    def test_estimator_contract(contract):
        estimator = CvarEstimator(alpha=0.2)
        assert contract(estimator)
"""

import numpy as np
from openfermion import IsingOperator
from zquantum.core.circuits import RX, RY, RZ, Circuit, H
from zquantum.core.interfaces.estimation import (
    EstimateExpectationValues,
    EstimationTask,
)
from zquantum.core.symbolic_simulator import SymbolicSimulator

_backend = SymbolicSimulator(seed=1997)

_estimation_tasks = [
    EstimationTask(IsingOperator("Z0"), Circuit([H(0)]), 10000),
    EstimationTask(
        IsingOperator("Z0") + IsingOperator("Z1") + IsingOperator("Z2"),
        Circuit([H(0), RX(np.pi / 3)(0), H(2)]),
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


def _validate_each_task_returns_one_expecation_value(
    estimator: EstimateExpectationValues,
):
    # When
    expectation_values = estimator(
        backend=_backend,
        estimation_tasks=_estimation_tasks,
    )

    # Then
    return len(expectation_values) == len(_estimation_tasks)


def _validate_order_of_outputs_matches_order_of_inputs(
    estimator: EstimateExpectationValues,
):
    expectation_values = estimator(
        backend=_backend,
        estimation_tasks=_estimation_tasks,
    )

    return all(
        [
            np.array_equal(
                expectation_values[i].values,
                estimator(
                    backend=_backend,
                    estimation_tasks=[task],
                )[0].values,
            )
            for i, task in enumerate(_estimation_tasks)
        ]
    )


def _validate_expectation_value_includes_coefficients(
    estimator: EstimateExpectationValues,
):
    estimation_tasks = [
        EstimationTask(IsingOperator("Z0"), Circuit([RX(np.pi / 3)(0)]), 10000),
        EstimationTask(
            IsingOperator("Z0", 19.971997), Circuit([RX(np.pi / 3)(0)]), 10000
        ),
    ]

    expectation_values = estimator(
        backend=_backend,
        estimation_tasks=estimation_tasks,
    )

    return not np.array_equal(
        expectation_values[0].values, expectation_values[1].values
    )


def _validate_constant_terms_are_included_in_output(
    estimator: EstimateExpectationValues,
):
    estimation_tasks = [
        EstimationTask(IsingOperator("Z0"), Circuit([H(0)]), 10000),
        EstimationTask(
            IsingOperator("Z0") + IsingOperator("[]", 19.971997),
            Circuit([H(0)]),
            10000,
        ),
    ]

    expectation_values = estimator(
        backend=_backend,
        estimation_tasks=estimation_tasks,
    )

    return not np.array_equal(
        expectation_values[0].values, expectation_values[1].values
    )


ESTIMATOR_CONTRACTS = [
    _validate_each_task_returns_one_expecation_value,
    _validate_order_of_outputs_matches_order_of_inputs,
    _validate_expectation_value_includes_coefficients,
    _validate_constant_terms_are_included_in_output,
]
