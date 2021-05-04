import numpy as np
import pytest
import sympy
from openfermion import IsingOperator, QubitOperator, qubit_operator_sparse
from pyquil import Program
from pyquil.gates import RX, RY, RZ, X, Y
from functools import partial
from zquantum.core.circuit import Circuit, Qubit, Gate
from zquantum.core.interfaces.mock_objects import (
    MockQuantumBackend,
    MockQuantumSimulator,
)
from zquantum.core.measurement import ExpectationValues
from zquantum.core.openfermion._utils import change_operator_type
from zquantum.core.wip.estimators.estimation import (
    calculate_exact_expectation_values,
    get_context_selection_circuit,
    get_context_selection_circuit_for_group,
    estimate_expectation_values_by_averaging,
    allocate_shots_proportionally,
    allocate_shots_uniformly,
    evaluate_estimation_circuits,
    group_greedily_with_context_selection,
    group_individually,
)
from zquantum.core.wip.estimators.estimation_interface import EstimationTask


class TestEstimatorUtils:
    def test_get_context_selection_circuit_offdiagonal(self):
        term = QubitOperator("X0 Y1")
        circuit, ising_operator = get_context_selection_circuit(term)

        # Need to convert to QubitOperator in order to get matrix representation
        qubit_operator = change_operator_type(ising_operator, QubitOperator)

        target_unitary = qubit_operator_sparse(term)
        transformed_unitary = (
            circuit.to_unitary().conj().T
            @ qubit_operator_sparse(qubit_operator)
            @ circuit.to_unitary()
        )
        assert np.allclose(target_unitary.todense(), transformed_unitary)

    def test_get_context_selection_circuit_diagonal(self):
        term = QubitOperator("Z2 Z4")
        circuit, ising_operator = get_context_selection_circuit(term)
        assert len(circuit.gates) == 0
        assert ising_operator == change_operator_type(term, IsingOperator)

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
        for circuit in circuits:
            circuit.qubits = [Qubit(i) for i in range(2)]

        circuits[1].gates = [
            Gate("Rx", [circuits[1].qubits[0]], [1.2]),
            Gate("Ry", [circuits[1].qubits[1]], [1.5]),
            Gate("Rx", [circuits[1].qubits[0]], [-0.0002]),
            Gate("Ry", [circuits[1].qubits[1]], [0]),
        ]

        for circuit in circuits[2:]:
            circuit.gates = [
                Gate("Rx", [circuit.qubits[0]], [sympy.Symbol("theta_0")]),
                Gate("Ry", [circuit.qubits[1]], [sympy.Symbol("theta_1")]),
                Gate("Rx", [circuit.qubits[0]], [sympy.Symbol("theta_2")]),
                Gate("Ry", [circuit.qubits[1]], [sympy.Symbol("theta_3")]),
            ]

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
            allocate_shots = allocate_shots_proportionally(
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
            assert new_task.circuit.symbolic_params == []

    def test_group_greedily_with_context_selection_all_different_groups(self):
        target_operator = 10.0 * QubitOperator("Z0")
        target_operator -= 3.0 * QubitOperator("Y0")
        target_operator += 1.0 * QubitOperator("X0")
        target_operator += 20.0 * QubitOperator("")

        expected_operator_terms_per_frame = [
            (10.0 * QubitOperator("Z0")).terms,
            (-3.0 * QubitOperator("Z0")).terms,
            (1.0 * QubitOperator("Z0")).terms,
            (20.0 * QubitOperator("")).terms,
        ]

        circuit = Circuit(Program(X(0)))

        estimation_tasks = [EstimationTask(target_operator, circuit, None)]

        grouped_tasks = group_greedily_with_context_selection(estimation_tasks)

        assert len(grouped_tasks) == 4

        for task in grouped_tasks:
            frame_circuit = circuit

            assert task.operator.terms in expected_operator_terms_per_frame
            expected_operator_terms_per_frame.remove(task.operator.terms)

            if task.operator.terms == (1.0 * QubitOperator("Z0")).terms:
                frame_circuit += Circuit(Program(RY(-np.pi / 2, 0)))
            elif task.operator.terms == (-3.0 * QubitOperator("Z0")).terms:
                frame_circuit += Circuit(Program(RX(np.pi / 2, 0)))
            assert frame_circuit == task.circuit
        assert len(expected_operator_terms_per_frame) == 0

    def test_group_greedily_with_context_selection_all_comeasureable(self):
        target_operator = 10.0 * QubitOperator("Y0")
        target_operator -= 3.0 * QubitOperator("Y0 Y1")
        target_operator += 1.0 * QubitOperator("Y1")
        target_operator += 20.0 * QubitOperator("Y0 Y1 Y2")

        expected_operator = 10.0 * QubitOperator("Z0")
        expected_operator -= 3.0 * QubitOperator("Z0 Z1")
        expected_operator += 1.0 * QubitOperator("Z1")
        expected_operator += 20.0 * QubitOperator("Z0 Z1 Z2")

        circuit = Circuit(Program([X(0), X(1), X(2)]))

        estimation_tasks = [EstimationTask(target_operator, circuit, None)]

        grouped_tasks = group_greedily_with_context_selection(estimation_tasks)

        assert len(grouped_tasks) == 1
        assert grouped_tasks[0].operator.terms == expected_operator.terms

        frame_circuit = circuit
        frame_circuit += Circuit(
            Program([RX(np.pi / 2, 0), RX(np.pi / 2, 1), RX(np.pi / 2, 2)])
        )
        assert grouped_tasks[0].circuit == frame_circuit

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

        circuit = Circuit(Program(X(0)))

        estimation_tasks = [EstimationTask(target_operator, circuit, None)]

        grouped_tasks = group_individually(estimation_tasks)

        assert len(grouped_tasks) == 5

        for task in grouped_tasks:
            assert task.operator.terms in expected_operator_terms_per_frame


class TestBasicEstimationMethods:
    @pytest.fixture()
    def backend(self):
        return MockQuantumBackend(n_samples=20)

    @pytest.fixture()
    def simulator(self):
        return MockQuantumSimulator()

    @pytest.fixture()
    def estimation_tasks(self):
        task_1 = EstimationTask(
            IsingOperator("Z0"), circuit=Circuit(Program(X(0))), number_of_shots=10
        )
        task_2 = EstimationTask(
            IsingOperator("Z0"),
            circuit=Circuit(Program(RZ(np.pi / 2, 0))),
            number_of_shots=20,
        )
        task_3 = EstimationTask(
            IsingOperator((), coefficient=2.0),
            circuit=Circuit(Program(RY(np.pi / 4, 0))),
            number_of_shots=30,
        )
        return [task_1, task_2, task_3]

    def test_estimate_expectation_values_by_averaging(self, backend, estimation_tasks):
        expectation_values_list = estimate_expectation_values_by_averaging(
            backend, estimation_tasks
        )
        assert len(expectation_values_list) == 3
        for expectation_values, task in zip(expectation_values_list, estimation_tasks):
            assert len(expectation_values.values) == len(task.operator.terms)

    def test_calculate_exact_expectation_values(self, simulator, estimation_tasks):
        expectation_values_list = calculate_exact_expectation_values(
            simulator, estimation_tasks
        )
        assert len(expectation_values_list) == 3
        for expectation_values, task in zip(expectation_values_list, estimation_tasks):
            assert len(expectation_values.values) == len(task.operator.terms)

    def test_calculate_exact_expectation_values_fails_with_non_simulator(
        self, estimation_tasks
    ):
        backend = MockQuantumBackend()
        with pytest.raises(AttributeError):
            expectation_values_list = calculate_exact_expectation_values(
                backend, estimation_tasks
            )