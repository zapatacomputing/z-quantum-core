"""Test case prototypes of instances of the EstimateExpectationValues protocol
that can be used in other projects.

Note that this file won't be executed on its own by pytest.
You need to define your own test cases that import the ones defined here.
Here is an example of how you would do that:

    from zquantum.core.interfaces.estimator_contract import ESTIMATOR_CONTRACT

    @pytest.mark.parametrize("contract", ESTIMATOR_CONTRACT)
    def test_estimator_contract(contract):
        estimator = CvarEstimator(alpha=0.2)
        assert contract(estimator)
"""

import numpy as np
from openfermion import IsingOperator
from zquantum.core.circuits import RX, RY, RZ, Circuit, H
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.symbolic_simulator import SymbolicSimulator

backend = SymbolicSimulator(seed=1997)

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

    return all(
        [
            expectation_values[i].values
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

    return expectation_values[0].values != expectation_values[1].values


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

    return expectation_values[0].values != expectation_values[1].values


ESTIMATOR_CONTRACT = [
    _validate_each_task_returns_one_expecation_value,
    _validate_order_of_outputs_matches_order_of_inputs,
    _validate_number_of_entries_in_each_expectation_value_is_not_restricted,
    _validate_expectation_value_includes_coefficients,
    _validate_constant_terms_are_included_in_output,
]
