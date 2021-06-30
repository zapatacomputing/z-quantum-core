from functools import partial

import numpy as np
import pytest
import sympy
from openfermion import IsingOperator, QubitOperator, qubit_operator_sparse
from openfermion.linalg.sparse_tools import (
    expectation_one_body_db_operator_computational_basis_state,
)
from zquantum.core.circuits import RX, RY, RZ, Circuit, H, X
from zquantum.core.estimation import (
    allocate_shots_proportionally,
    allocate_shots_uniformly,
    calculate_exact_expectation_values,
    estimate_expectation_values_by_averaging,
    evaluate_constant_estimation_tasks,
    evaluate_estimation_circuits,
    get_context_selection_circuit_for_group,
    group_greedily,
    group_individually,
    perform_context_selection,
    split_constant_estimation_tasks,
)
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.interfaces.mock_objects import MockQuantumBackend
from zquantum.core.measurement import ExpectationValues
from zquantum.core.openfermion._utils import change_operator_type
from zquantum.core.symbolic_simulator import SymbolicSimulator


class TestEstimatorUtils:
    def test_get_context_selection_circuit_for_group(self):
        group = QubitOperator("X0 Y1") - 0.5 * QubitOperator((1, "Y"))
        circuit, ising_operator = get_context_selection_circuit_for_group(group)

        # Need to convert to QubitOperator in order to get matrix representation
        qubit_operator = change_operator_type(ising_operator, QubitOperator)

        target_unitary = qubit_operator_sparse(group)
        transformed_unitary = (
            circuit.to_unitary().conj().T
            @ qubit_operator_sparse(qubit_operator)
            @ circuit.to_unitary()
        )

        assert np.allclose(target_unitary.todense(), transformed_unitary)

    def test_perform_context_selection(self):
        target_operators = []
        target_operators.append(10.0 * QubitOperator("Z0"))
        target_operators.append(-3 * QubitOperator("Y0"))
        target_operators.append(1 * QubitOperator("X0"))
        target_operators.append(20 * QubitOperator(""))

        expected_operators = []
        expected_operators.append(10.0 * QubitOperator("Z0"))
        expected_operators.append(-3 * QubitOperator("Z0"))
        expected_operators.append(1 * QubitOperator("Z0"))
        expected_operators.append(20 * QubitOperator(""))

        base_circuit = Circuit([X(0)])
        x_term_circuit = Circuit([RY(-np.pi / 2)(0)])
        y_term_circuit = Circuit([RX(np.pi / 2)(0)])

        expected_circuits = [
            base_circuit,
            base_circuit + y_term_circuit,
            base_circuit + x_term_circuit,
            base_circuit,
        ]

        estimation_tasks = [
            EstimationTask(operator, base_circuit, None)
            for operator in target_operators
        ]

        tasks_with_context_selection = perform_context_selection(estimation_tasks)

        for task, expected_circuit, expected_operator in zip(
            tasks_with_context_selection, expected_circuits, expected_operators
        ):
            assert task.operator.terms == expected_operator.terms
            assert task.circuit == expected_circuit

    @pytest.fixture()
    def frame_operators(self):
        operators = [
            2.0 * IsingOperator((1, "Z")) * IsingOperator((2, "Z")),
            1.0 * IsingOperator((3, "Z")) * IsingOperator((0, "Z")),
            -1.0 * IsingOperator((2, "Z")),
        ]

        return operators

    @pytest.fixture()
    def circuits(self):
        circuits = [Circuit() for _ in range(5)]

        circuits[1] += RX(1.2)(0)
        circuits[1] += RY(1.5)(1)
        circuits[1] += RX(-0.0002)(0)
        circuits[1] += RY(0)(1)

        for circuit in circuits[2:]:
            circuit += RX(sympy.Symbol("theta_0"))(0)
            circuit += RY(sympy.Symbol("theta_1"))(1)
            circuit += RX(sympy.Symbol("theta_2"))(0)
            circuit += RY(sympy.Symbol("theta_3"))(1)

        return circuits

    @pytest.mark.parametrize(
        "n_samples, target_n_samples_list",
        [
            (100, [100, 100, 100]),
            (17, [17, 17, 17]),
        ],
    )
    def test_allocate_shots_uniformly(
        self,
        frame_operators,
        n_samples,
        target_n_samples_list,
    ):
        allocate_shots = partial(allocate_shots_uniformly, number_of_shots=n_samples)
        circuit = Circuit()
        estimation_tasks = [
            EstimationTask(operator, circuit, 1) for operator in frame_operators
        ]

        new_estimation_tasks = allocate_shots(estimation_tasks)

        for task, target_n_samples in zip(new_estimation_tasks, target_n_samples_list):
            assert task.number_of_shots == target_n_samples

    @pytest.mark.parametrize(
        "total_n_shots, prior_expectation_values, target_n_samples_list",
        [
            (400, None, [200, 100, 100]),
            (400, ExpectationValues(np.array([0, 0, 0])), [200, 100, 100]),
            (400, ExpectationValues(np.array([1, 0.3, 0.3])), [0, 200, 200]),
        ],
    )
    def test_allocate_shots_proportionally(
        self,
        frame_operators,
        total_n_shots,
        prior_expectation_values,
        target_n_samples_list,
    ):
        allocate_shots = partial(
            allocate_shots_proportionally,
            total_n_shots=total_n_shots,
            prior_expectation_values=prior_expectation_values,
        )
        circuit = Circuit()
        estimation_tasks = [
            EstimationTask(operator, circuit, 1) for operator in frame_operators
        ]

        new_estimation_tasks = allocate_shots(estimation_tasks)

        for task, target_n_samples in zip(new_estimation_tasks, target_n_samples_list):
            assert task.number_of_shots == target_n_samples

    @pytest.mark.parametrize(
        "n_samples",
        [-1],
    )
    def test_allocate_shots_uniformly_invalid_inputs(
        self,
        n_samples,
    ):
        estimation_tasks = []
        with pytest.raises(ValueError):
            allocate_shots_uniformly(estimation_tasks, number_of_shots=n_samples)

    @pytest.mark.parametrize(
        "total_n_shots, prior_expectation_values",
        [
            (-1, ExpectationValues(np.array([0, 0, 0]))),
        ],
    )
    def test_allocate_shots_proportionally_invalid_inputs(
        self,
        total_n_shots,
        prior_expectation_values,
    ):
        estimation_tasks = []
        with pytest.raises(ValueError):
            _ = allocate_shots_proportionally(
                estimation_tasks, total_n_shots, prior_expectation_values
            )

    def test_evaluate_estimation_circuits_no_symbols(
        self,
        circuits,
    ):
        evaluate_circuits = partial(
            evaluate_estimation_circuits, symbols_maps=[[] for _ in circuits]
        )
        operator = QubitOperator()
        estimation_tasks = [
            EstimationTask(operator, circuit, 1) for circuit in circuits
        ]

        new_estimation_tasks = evaluate_circuits(estimation_tasks)

        for old_task, new_task in zip(estimation_tasks, new_estimation_tasks):
            assert old_task.circuit == new_task.circuit

    def test_evaluate_estimation_circuits_all_symbols(
        self,
        circuits,
    ):
        symbols_maps = [
            [
                (sympy.Symbol("theta_0"), 0),
                (sympy.Symbol("theta_1"), 0),
                (sympy.Symbol("theta_2"), 0),
                (sympy.Symbol("theta_3"), 0),
            ]
            for _ in circuits
        ]
        evaluate_circuits = partial(
            evaluate_estimation_circuits,
            symbols_maps=symbols_maps,
        )
        operator = QubitOperator()
        estimation_tasks = [
            EstimationTask(operator, circuit, 1) for circuit in circuits
        ]

        new_estimation_tasks = evaluate_circuits(estimation_tasks)

        for new_task in new_estimation_tasks:
            assert len(new_task.circuit.free_symbols) == 0

    def test_group_greedily_all_different_groups(self):
        target_operator = 10.0 * QubitOperator("Z0")
        target_operator -= 3.0 * QubitOperator("Y0")
        target_operator += 1.0 * QubitOperator("X0")
        target_operator += 20.0 * QubitOperator("")

        expected_operators = [
            10.0 * QubitOperator("Z0"),
            -3.0 * QubitOperator("Y0"),
            1.0 * QubitOperator("X0"),
            20.0 * QubitOperator(""),
        ]

        circuit = Circuit([X(0)])

        estimation_tasks = [EstimationTask(target_operator, circuit, None)]

        grouped_tasks = group_greedily(estimation_tasks)

        for task, operator in zip(grouped_tasks, expected_operators):
            assert task.operator == operator

        for initial_task, modified_task in zip(estimation_tasks, grouped_tasks):
            assert modified_task.circuit == initial_task.circuit
            assert modified_task.number_of_shots == initial_task.number_of_shots

    def test_group_greedily_all_comeasureable(self):
        target_operator = 10.0 * QubitOperator("Y0")
        target_operator -= 3.0 * QubitOperator("Y0 Y1")
        target_operator += 1.0 * QubitOperator("Y1")
        target_operator += 20.0 * QubitOperator("Y0 Y1 Y2")

        circuit = Circuit([X(0), X(1), X(2)])

        estimation_tasks = [EstimationTask(target_operator, circuit, None)]

        grouped_tasks = group_greedily(estimation_tasks)

        assert len(grouped_tasks) == 1
        assert grouped_tasks[0].operator == target_operator

        for initial_task, modified_task in zip(estimation_tasks, grouped_tasks):
            assert modified_task.circuit == initial_task.circuit
            assert modified_task.number_of_shots == initial_task.number_of_shots

    def test_group_individually(self):
        target_operator = 10.0 * QubitOperator("Z0")
        target_operator += 5.0 * QubitOperator("Z1")
        target_operator -= 3.0 * QubitOperator("Y0")
        target_operator += 1.0 * QubitOperator("X0")
        target_operator += 20.0 * QubitOperator("")

        expected_operator_terms_per_frame = [
            (10.0 * QubitOperator("Z0")).terms,
            (5.0 * QubitOperator("Z1")).terms,
            (-3.0 * QubitOperator("Y0")).terms,
            (1.0 * QubitOperator("X0")).terms,
            (20.0 * QubitOperator("")).terms,
        ]

        circuit = Circuit([X(0)])

        estimation_tasks = [EstimationTask(target_operator, circuit, None)]

        grouped_tasks = group_individually(estimation_tasks)

        assert len(grouped_tasks) == 5

        for task in grouped_tasks:
            assert task.operator.terms in expected_operator_terms_per_frame

    @pytest.mark.parametrize(
        ",".join(
            [
                "estimation_tasks",
                "ref_estimation_tasks_to_measure",
                "ref_constant_estimation_tasks",
                "ref_indices_to_measure",
                "ref_constant_indices",
            ]
        ),
        [
            (
                [
                    EstimationTask(
                        IsingOperator("2[Z0] + 3 [Z1 Z2]"), Circuit([X(0)]), 10
                    ),
                    EstimationTask(
                        IsingOperator("2[Z0] + 3 [Z1 Z2] + 4[]"),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        IsingOperator("4[Z3]"),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [
                    EstimationTask(
                        IsingOperator("2[Z0] + 3 [Z1 Z2]"), Circuit([X(0)]), 10
                    ),
                    EstimationTask(
                        IsingOperator("2[Z0] + 3 [Z1 Z2] + 4 []"),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        IsingOperator("4[Z3]"),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [],
                [0, 1, 2],
                [],
            ),
            (
                [
                    EstimationTask(
                        IsingOperator("2[Z0] + 3 [Z1 Z2]"), Circuit([X(0)]), 10
                    ),
                    EstimationTask(
                        IsingOperator("4[] "),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        IsingOperator("4[Z3]"),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [
                    EstimationTask(
                        IsingOperator("2[Z0] + 3 [Z1 Z2]"), Circuit([X(0)]), 10
                    ),
                    EstimationTask(
                        IsingOperator("4[Z3]"),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [
                    EstimationTask(
                        IsingOperator("4[]"), Circuit([RZ(np.pi / 2)(0)]), 1000
                    )
                ],
                [0, 2],
                [1],
            ),
            (
                [
                    EstimationTask(IsingOperator("- 3 []"), Circuit([X(0)]), 0),
                    EstimationTask(
                        IsingOperator("2[Z0] + 3 [Z1 Z2] + 4[]"),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        IsingOperator("4[Z3]"),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [
                    EstimationTask(
                        IsingOperator("2[Z0] + 3 [Z1 Z2] + 4 []"),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                    EstimationTask(
                        IsingOperator("4[Z3]"),
                        Circuit([RY(np.pi / 2)(0)]),
                        17,
                    ),
                ],
                [
                    EstimationTask(IsingOperator("- 3 []"), Circuit([X(0)]), 0),
                ],
                [1, 2],
                [0],
            ),
        ],
    )
    def test_split_constant_estimation_tasks(
        self,
        estimation_tasks,
        ref_estimation_tasks_to_measure,
        ref_constant_estimation_tasks,
        ref_indices_to_measure,
        ref_constant_indices,
    ):

        (
            estimation_task_to_measure,
            constant_estimation_tasks,
            indices_to_measure,
            indices_for_constants,
        ) = split_constant_estimation_tasks(estimation_tasks)

        assert estimation_task_to_measure == ref_estimation_tasks_to_measure
        assert constant_estimation_tasks == ref_constant_estimation_tasks
        assert indices_to_measure == ref_indices_to_measure
        assert ref_constant_indices == indices_for_constants

    @pytest.mark.parametrize(
        "estimation_tasks",
        [
            [
                EstimationTask(IsingOperator("- 3 []"), Circuit([X(0)]), 0),
                EstimationTask(
                    IsingOperator("2[Z0] + 3 [Z1 Z2] + 4[]"),
                    Circuit([RZ(np.pi / 2)(0)]),
                    0,
                ),
                EstimationTask(
                    IsingOperator("4[Z3]"),
                    Circuit([RY(np.pi / 2)(0)]),
                    17,
                ),
            ],
        ],
    )
    def test_split_constant_estimation_tasks_fails_with_zero_shots(
        self, estimation_tasks
    ):

        with pytest.raises(RuntimeError):
            _ = split_constant_estimation_tasks(estimation_tasks)

    @pytest.mark.parametrize(
        "estimation_tasks,ref_expectation_values",
        [
            (
                [
                    EstimationTask(
                        IsingOperator("4[] "),
                        Circuit([RZ(np.pi / 2)(0)]),
                        1000,
                    ),
                ],
                [
                    ExpectationValues(
                        np.asarray([4.0]),
                        correlations=[np.asarray([[0.0]])],
                        estimator_covariances=[np.asarray([[0.0]])],
                    ),
                ],
            ),
            (
                [
                    EstimationTask(
                        IsingOperator("- 2.5 [] - 0.5 []"), Circuit([X(0)]), 0
                    ),
                    EstimationTask(
                        IsingOperator("0.001[] "), Circuit([RZ(np.pi / 2)(0)]), 2
                    ),
                ],
                [
                    ExpectationValues(
                        np.asarray([-3.0]),
                        correlations=[np.asarray([[0.0]])],
                        estimator_covariances=[np.asarray([[0.0]])],
                    ),
                    ExpectationValues(
                        np.asarray([0.001]),
                        correlations=[np.asarray([[0.0]])],
                        estimator_covariances=[np.asarray([[0.0]])],
                    ),
                ],
            ),
        ],
    )
    def test_evaluate_constant_estimation_tasks(
        self, estimation_tasks, ref_expectation_values
    ):

        expectation_values = evaluate_constant_estimation_tasks(estimation_tasks)

        for ex_val, ref_ex_val in zip(expectation_values, ref_expectation_values):
            assert np.allclose(ex_val.values, ref_ex_val.values)
            assert np.allclose(ex_val.correlations, ref_ex_val.correlations)
            assert np.allclose(
                ex_val.estimator_covariances, ref_ex_val.estimator_covariances
            )

    @pytest.mark.parametrize(
        "estimation_tasks",
        [
            (
                [
                    EstimationTask(
                        IsingOperator("- 2.5 [] - 0.5 [Z1]"), Circuit([X(0)]), 0
                    ),
                ]
            ),
            (
                [
                    EstimationTask(
                        IsingOperator("0.001 [Z0]"),
                        Circuit([RZ(np.pi / 2)(0)]),
                        0,
                    ),
                    EstimationTask(
                        IsingOperator("2.0[] "), Circuit([RZ(np.pi / 2)(0)]), 2
                    ),
                ]
            ),
        ],
    )
    def test_evaluate_constant_estimation_tasks_fails_with_non_constant(
        self, estimation_tasks
    ):
        with pytest.raises(RuntimeError):
            _ = evaluate_constant_estimation_tasks(estimation_tasks)


TEST_CASES_EIGENSTATES = [
    (
        [
            EstimationTask(
                IsingOperator("Z0"), circuit=Circuit([X(0)]), number_of_shots=10
            ),
            EstimationTask(
                IsingOperator((), coefficient=2.0),
                circuit=Circuit([RY(np.pi / 4)(0)]),
                number_of_shots=30,
            ),
        ],
        [ExpectationValues(np.array([-1])), ExpectationValues(np.array([2]))],
    ),
]
TEST_CASES_NONEIGENSTATES = [
    (
        [
            EstimationTask(
                IsingOperator("Z0"),
                circuit=Circuit([H(0)]),
                number_of_shots=1000,
            ),
            EstimationTask(
                IsingOperator("Z0", coefficient=-2),
                circuit=Circuit([RY(np.pi / 4)(0)]),
                number_of_shots=1000,
            ),
        ],
        [
            ExpectationValues(np.array([0])),
            ExpectationValues(np.array([-2 * (np.cos(np.pi / 8) ** 2 - 0.5) * 2])),
        ],
    ),
]


class TestBasicEstimationMethods:
    @pytest.fixture()
    def simulator(self):
        return SymbolicSimulator()

    @pytest.fixture()
    def estimation_tasks(self):
        task_1 = EstimationTask(
            IsingOperator("Z0"), circuit=Circuit([X(0)]), number_of_shots=10
        )
        task_2 = EstimationTask(
            IsingOperator("Z0"),
            circuit=Circuit([RY(np.pi / 2)(0)]),
            number_of_shots=20,
        )
        task_3 = EstimationTask(
            IsingOperator((), coefficient=2.0),
            circuit=Circuit([RY(np.pi / 4)(0)]),
            number_of_shots=30,
        )
        return [task_1, task_2, task_3]

    @pytest.fixture()
    def target_expectation_values(self):
        return [ExpectationValues(-1), ExpectationValues(0), ExpectationValues(2)]

    @pytest.mark.parametrize(
        "estimation_tasks,target_expectations", TEST_CASES_EIGENSTATES
    )
    def test_estimate_expectation_values_by_averaging_for_eigenstates(
        self, simulator, estimation_tasks, target_expectations
    ):
        expectation_values_list = estimate_expectation_values_by_averaging(
            simulator, estimation_tasks
        )
        for expectation_values, target, task in zip(
            expectation_values_list, target_expectations, estimation_tasks
        ):
            assert len(expectation_values.values) == len(task.operator.terms)
            # TODO: add tests for correlations and covariances
            np.testing.assert_array_equal(expectation_values.values, target.values)

    @pytest.mark.parametrize(
        "estimation_tasks,target_expectations", TEST_CASES_NONEIGENSTATES
    )
    def test_estimate_expectation_values_by_averaging_for_non_eigenstates(
        self, simulator, estimation_tasks, target_expectations
    ):

        expectation_values_list = estimate_expectation_values_by_averaging(
            simulator, estimation_tasks
        )
        for expectation_values, target, task in zip(
            expectation_values_list, target_expectations, estimation_tasks
        ):
            assert len(expectation_values.values) == len(task.operator.terms)
            # TODO: add tests for correlations and covariances
            np.testing.assert_allclose(
                expectation_values.values, target.values, atol=0.1
            )

    @pytest.mark.parametrize(
        "estimation_tasks,target_expectations",
        TEST_CASES_EIGENSTATES + TEST_CASES_NONEIGENSTATES,
    )
    def test_calculate_exact_expectation_values(
        self, simulator, estimation_tasks, target_expectations
    ):
        expectation_values_list = calculate_exact_expectation_values(
            simulator, estimation_tasks
        )
        for expectation_values, target, task in zip(
            expectation_values_list, target_expectations, estimation_tasks
        ):
            assert len(expectation_values.values) == len(task.operator.terms)
            # TODO: add tests for correlations and covariances
            np.testing.assert_array_almost_equal(
                expectation_values.values, target.values
            )

    def test_calculate_exact_expectation_values_fails_with_non_simulator(
        self, estimation_tasks
    ):
        backend = MockQuantumBackend()
        with pytest.raises(AttributeError):
            _ = calculate_exact_expectation_values(backend, estimation_tasks)
