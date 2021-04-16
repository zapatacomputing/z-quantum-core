import numpy as np
import pytest
from openfermion import IsingOperator, QubitOperator, qubit_operator_sparse
from pyquil import Program
from pyquil.gates import RY, RZ, X
from functools import partial
from zquantum.core.circuit import Circuit
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
    naively_estimate_expectation_values,
    proportional_shot_allocation,
    uniform_shot_allocation,
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

    @pytest.mark.parametrize(
        "n_samples, target_n_samples_list",
        [
            (100, [100, 100, 100]),
            (17, [17, 17, 17]),
        ],
    )
    def test_uniform_shot_allocation(
        self,
        frame_operators,
        n_samples,
        target_n_samples_list,
    ):
        allocate_shots = partial(uniform_shot_allocation, number_of_shots=n_samples)
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
    def test_proportional_shot_allocation(
        self,
        frame_operators,
        total_n_shots,
        prior_expectation_values,
        target_n_samples_list,
    ):
        allocate_shots = partial(
            proportional_shot_allocation,
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
    def test_uniform_shot_allocation_invalid_inputs(
        self,
        n_samples,
    ):
        estimation_tasks = []
        with pytest.raises(ValueError):
            uniform_shot_allocation(estimation_tasks, number_of_shots=n_samples)

    @pytest.mark.parametrize(
        "total_n_shots, prior_expectation_values",
        [
            (-1, ExpectationValues(np.array([0, 0, 0]))),
        ],
    )
    def test_proportional_shot_allocation_invalid_inputs(
        self,
        total_n_shots,
        prior_expectation_values,
    ):
        estimation_tasks = []
        with pytest.raises(ValueError):
            allocate_shots = proportional_shot_allocation(
                estimation_tasks, total_n_shots, prior_expectation_values
            )


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

    def test_naively_estimate_expectation_values(self, backend, estimation_tasks):
        expectation_values = naively_estimate_expectation_values(
            backend, estimation_tasks
        )
        assert len(expectation_values.values) == 3

    def test_calculate_exact_expectation_values(self, simulator, estimation_tasks):
        expectation_values = calculate_exact_expectation_values(
            simulator, estimation_tasks
        )
        assert len(expectation_values.values) == 3

    def test_calculate_exact_expectation_values_fails_with_non_simulator(
        self, estimation_tasks
    ):
        backend = MockQuantumBackend()
        with pytest.raises(AttributeError):
            expectation_values = calculate_exact_expectation_values(
                backend, estimation_tasks
            )