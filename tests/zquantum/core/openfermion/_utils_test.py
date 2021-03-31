import unittest
import random
import numpy as np
import pyquil
import os

from cirq import GridQubit, LineQubit, X, Y, Z, PauliSum, PauliString
from openfermion import (
    QubitOperator,
    IsingOperator,
    FermionOperator,
    qubit_operator_sparse,
    get_interaction_operator,
    get_fermion_operator,
    jordan_wigner,
    get_sparse_operator,
)
from openfermion.hamiltonians import fermi_hubbard
from openfermion.linalg import jw_get_ground_state_at_particle_number

from ..circuit import Circuit, Gate, Qubit, build_uniform_param_grid
from ..measurement import ExpectationValues
from ..utils import RNDSEED, create_object, hf_rdm
from ..interfaces.mock_objects import MockAnsatz

from ._io import load_interaction_operator

from ._utils import (
    generate_random_qubitop,
    get_qubitop_from_coeffs_and_labels,
    evaluate_qubit_operator,
    get_qubitop_from_matrix,
    reverse_qubit_order,
    get_expectation_value,
    change_operator_type,
    evaluate_operator_for_parameter_grid,
    get_fermion_number_operator,
    get_diagonal_component,
    get_polynomial_tensor,
    qubitop_to_paulisum,
    create_circuits_from_qubit_operator,
    evaluate_qubit_operator_list,
    get_ground_state_rdm_from_qubit_op,
    remove_inactive_orbitals,
)


class TestQubitOperator(unittest.TestCase):
    def test_build_qubitoperator_from_coeffs_and_labels(self):
        # Given
        test_op = QubitOperator(((0, "Y"), (1, "X"), (2, "Z"), (4, "X")), 3.0j)
        coeffs = [3.0j]
        labels = [[2, 1, 3, 0, 1]]

        # When
        build_op = get_qubitop_from_coeffs_and_labels(coeffs, labels)

        # Then
        self.assertEqual(test_op, build_op)

    def test_qubitop_matrix_converion(self):
        # Given
        m = 4
        n = 2 ** m
        TOL = 10 ** -15
        random.seed(RNDSEED)
        A = np.array([[random.uniform(-1, 1) for x in range(n)] for y in range(n)])

        # When
        A_qubitop = get_qubitop_from_matrix(A)
        A_qubitop_matrix = np.array(qubit_operator_sparse(A_qubitop).todense())
        test_matrix = A_qubitop_matrix - A

        # Then
        for row in test_matrix:
            for elem in row:
                self.assertEqual(abs(elem) < TOL, True)

    def test_generate_random_qubitop(self):
        # Given
        nqubits = 4
        nterms = 5
        nlocality = 2
        max_coeff = 1.5
        fixed_coeff = False

        # When
        qubit_op = generate_random_qubitop(
            nqubits, nterms, nlocality, max_coeff, fixed_coeff
        )
        # Then
        self.assertEqual(len(qubit_op.terms), nterms)
        for term, coefficient in qubit_op.terms.items():
            for i in range(nlocality):
                self.assertLess(term[i][0], nqubits)
            self.assertEqual(len(term), nlocality)
            self.assertLessEqual(np.abs(coefficient), max_coeff)

        # Given
        fixed_coeff = True
        # When
        qubit_op = generate_random_qubitop(
            nqubits, nterms, nlocality, max_coeff, fixed_coeff
        )
        # Then
        self.assertEqual(len(qubit_op.terms), nterms)
        for term, coefficient in qubit_op.terms.items():
            self.assertEqual(np.abs(coefficient), max_coeff)

    def test_evaluate_qubit_operator(self):
        # Given
        qubit_op = QubitOperator("0.5 [] + 0.5 [Z1]")
        expectation_values = ExpectationValues([0.5, 0.5])
        # When
        value_estimate = evaluate_qubit_operator(qubit_op, expectation_values)
        # Then
        self.assertAlmostEqual(value_estimate.value, 0.5)

    def test_evaluate_qubit_operator_list(self):
        # Given
        qubit_op_list = [
            QubitOperator("0.5 [] + 0.5 [Z1]"),
            QubitOperator("0.3 [X1] + 0.2[Y2]"),
        ]
        expectation_values = ExpectationValues([0.5, 0.5, 0.4, 0.6])
        # When
        value_estimate = evaluate_qubit_operator_list(qubit_op_list, expectation_values)
        # Then
        self.assertAlmostEqual(value_estimate.value, 0.74)

    def test_evaluate_operator_for_parameter_grid(self):
        # Given
        ansatz = MockAnsatz(4, 2)
        grid = build_uniform_param_grid(1, 2, 0, np.pi, np.pi / 10)
        backend = create_object(
            {
                "module_name": "zquantum.core.interfaces.mock_objects",
                "function_name": "MockQuantumSimulator",
            }
        )
        op = QubitOperator("0.5 [] + 0.5 [Z1]")
        previous_layer_parameters = [1, 1]
        # When
        (
            parameter_grid_evaluation,
            optimal_parameters,
        ) = evaluate_operator_for_parameter_grid(
            ansatz, grid, backend, op, previous_layer_params=previous_layer_parameters
        )
        # Then (for brevity, only check first and last evaluations)
        self.assertIsInstance(parameter_grid_evaluation[0]["value"].value, float)
        self.assertEqual(parameter_grid_evaluation[0]["parameter1"], 0)
        self.assertEqual(parameter_grid_evaluation[0]["parameter2"], 0)
        self.assertIsInstance(parameter_grid_evaluation[99]["value"].value, float)
        self.assertEqual(
            parameter_grid_evaluation[99]["parameter1"], np.pi - np.pi / 10
        )
        self.assertEqual(
            parameter_grid_evaluation[99]["parameter2"], np.pi - np.pi / 10
        )

        self.assertEqual(len(optimal_parameters), 4)
        self.assertEqual(optimal_parameters[0], 1)
        self.assertEqual(optimal_parameters[1], 1)

    def test_reverse_qubit_order(self):
        # Given
        op1 = QubitOperator("[Z0 Z1]")
        op2 = QubitOperator("[Z1 Z0]")

        # When/Then
        self.assertEqual(op1, reverse_qubit_order(op2))

        # Given
        op1 = QubitOperator("Z0")
        op2 = QubitOperator("Z1")

        # When/Then
        self.assertEqual(op1, reverse_qubit_order(op2, n_qubits=2))
        self.assertEqual(op2, reverse_qubit_order(op1, n_qubits=2))

    def test_get_expectation_value(self):
        """Check <Z0> and <Z1> for the state |100>"""
        # Given
        wf = pyquil.wavefunction.Wavefunction([0, 1, 0, 0, 0, 0, 0, 0])
        op1 = QubitOperator("Z0")
        op2 = QubitOperator("Z1")
        # When
        exp_op1 = get_expectation_value(op1, wf)
        exp_op2 = get_expectation_value(op2, wf)

        # Then
        self.assertAlmostEqual(-1, exp_op1)
        self.assertAlmostEqual(1, exp_op2)

    def test_change_operator_type(self):
        # Given
        operator1 = QubitOperator("Z0 Z1", 4.5)
        operator2 = IsingOperator("Z0 Z1", 4.5)
        operator3 = IsingOperator()
        operator4 = IsingOperator("Z0", 0.5) + IsingOperator("Z1", 2.5)
        # When
        new_operator1 = change_operator_type(operator1, IsingOperator)
        new_operator2 = change_operator_type(operator2, QubitOperator)
        new_operator3 = change_operator_type(operator3, QubitOperator)
        new_operator4 = change_operator_type(operator4, QubitOperator)

        # Then
        self.assertEqual(IsingOperator("Z0 Z1", 4.5), new_operator1)
        self.assertEqual(QubitOperator("Z0 Z1", 4.5), new_operator2)
        self.assertEqual(QubitOperator(), new_operator3)
        self.assertEqual(
            QubitOperator("Z0", 0.5) + QubitOperator("Z1", 2.5), new_operator4
        )

    def test_get_fermion_number_operator(self):
        # Given
        n_qubits = 4
        n_particles = None
        correct_operator = get_interaction_operator(
            FermionOperator(
                """
        0.0 [] +
        1.0 [0^ 0] +
        1.0 [1^ 1] +
        1.0 [2^ 2] +
        1.0 [3^ 3]
        """
            )
        )

        # When
        number_operator = get_fermion_number_operator(n_qubits)

        # Then
        self.assertEqual(number_operator, correct_operator)

        # Given
        n_qubits = 4
        n_particles = 2
        correct_operator = get_interaction_operator(
            FermionOperator(
                """
        -2.0 [] +
        1.0 [0^ 0] +
        1.0 [1^ 1] +
        1.0 [2^ 2] +
        1.0 [3^ 3]
        """
            )
        )

        # When
        number_operator = get_fermion_number_operator(n_qubits, n_particles)

        # Then
        self.assertEqual(number_operator, correct_operator)

    def test_create_circuits_from_qubit_operator(self):
        # Initialize target
        qubits = [Qubit(i) for i in range(0, 2)]

        gate_Z0 = Gate("Z", [qubits[0]])
        gate_X1 = Gate("X", [qubits[1]])

        gate_Y0 = Gate("Y", [qubits[0]])
        gate_Z1 = Gate("Z", [qubits[1]])

        circuit1 = Circuit()
        circuit1.qubits = qubits
        circuit1.gates = [gate_Z0, gate_X1]

        circuit2 = Circuit()
        circuit2.qubits = qubits
        circuit2.gates = [gate_Y0, gate_Z1]

        target_circuits_list = [circuit1, circuit2]

        # Given
        qubit_op = QubitOperator("Z0 X1") + QubitOperator("Y0 Z1")

        # When
        pauli_circuits = create_circuits_from_qubit_operator(qubit_op)

        # Then
        self.assertEqual(pauli_circuits[0].gates, target_circuits_list[0].gates)
        self.assertEqual(pauli_circuits[1].gates, target_circuits_list[1].gates)
        self.assertEqual(
            str(pauli_circuits[0].qubits), str(target_circuits_list[0].qubits)
        )
        self.assertEqual(
            str(pauli_circuits[1].qubits), str(target_circuits_list[1].qubits)
        )


class TestOtherUtils(unittest.TestCase):
    def test_get_diagonal_component_polynomial_tensor(self):
        fermion_op = FermionOperator("0^ 1^ 2^ 0 1 2", 1.0)
        fermion_op += FermionOperator("0^ 1^ 2^ 0 1 3", 2.0)
        fermion_op += FermionOperator((), 3.0)
        polynomial_tensor = get_polynomial_tensor(fermion_op)
        diagonal_op, remainder_op = get_diagonal_component(polynomial_tensor)
        self.assertTrue((diagonal_op + remainder_op) == polynomial_tensor)
        diagonal_qubit_op = jordan_wigner(get_fermion_operator(diagonal_op))
        remainder_qubit_op = jordan_wigner(get_fermion_operator(remainder_op))
        for term in diagonal_qubit_op.terms:
            for pauli in term:
                self.assertTrue(pauli[1] == "Z")
        for term in remainder_qubit_op.terms:
            is_diagonal = True
            for pauli in term:
                if pauli[1] != "Z":
                    is_diagonal = False
                    break
            self.assertFalse(is_diagonal)

    def test_get_diagonal_component_interaction_op(self):
        fermion_op = FermionOperator("1^ 1", 0.5)
        fermion_op += FermionOperator("2^ 2", 0.5)
        fermion_op += FermionOperator("1^ 2^ 0 3", 0.5)
        diagonal_op, remainder_op = get_diagonal_component(
            get_interaction_operator(fermion_op)
        )
        self.assertTrue(
            (diagonal_op + remainder_op) == get_interaction_operator(fermion_op)
        )
        diagonal_qubit_op = jordan_wigner(diagonal_op)
        remainder_qubit_op = jordan_wigner(remainder_op)
        for term in diagonal_qubit_op.terms:
            for pauli in term:
                self.assertTrue(pauli[1] == "Z")
        is_diagonal = True
        for term in remainder_qubit_op.terms:
            for pauli in term:
                if pauli[1] != "Z":
                    is_diagonal = False
                    break
        self.assertFalse(is_diagonal)

    def test_qubitop_to_paulisum_identity_operator(self):
        # Given
        qubit_operator = QubitOperator("", 4)

        # When
        paulisum = qubitop_to_paulisum(qubit_operator)

        # Then
        self.assertEqual(paulisum.qubits, ())
        self.assertEqual(paulisum, PauliSum() + 4)

    def test_qubitop_to_paulisum_z0z1_operator(self):
        # Given
        qubit_operator = QubitOperator("Z0 Z1", -1.5)
        expected_qubits = (GridQubit(0, 0), GridQubit(1, 0))
        expected_paulisum = (
            PauliSum()
            + PauliString(Z.on(expected_qubits[0]))
            * PauliString(Z.on(expected_qubits[1]))
            * -1.5
        )

        # When
        paulisum = qubitop_to_paulisum(qubit_operator)

        # Then
        self.assertEqual(paulisum.qubits, expected_qubits)
        self.assertEqual(paulisum, expected_paulisum)

    def test_qubitop_to_paulisum_setting_qubits(self):
        # Given
        qubit_operator = QubitOperator("Z0 Z1", -1.5)
        expected_qubits = (LineQubit(0), LineQubit(5))
        expected_paulisum = (
            PauliSum()
            + PauliString(Z.on(expected_qubits[0]))
            * PauliString(Z.on(expected_qubits[1]))
            * -1.5
        )

        # When
        paulisum = qubitop_to_paulisum(qubit_operator, qubits=expected_qubits)

        # Then
        self.assertEqual(paulisum.qubits, expected_qubits)
        self.assertEqual(paulisum, expected_paulisum)

    def test_qubitop_to_paulisum_more_terms(self):
        # Given
        qubit_operator = (
            QubitOperator("Z0 Z1 Z2", -1.5)
            + QubitOperator("X0", 2.5)
            + QubitOperator("Y1", 3.5)
        )
        expected_qubits = (LineQubit(0), LineQubit(5), LineQubit(8))
        expected_paulisum = (
            PauliSum()
            + (
                PauliString(Z.on(expected_qubits[0]))
                * PauliString(Z.on(expected_qubits[1]))
                * PauliString(Z.on(expected_qubits[2]))
                * -1.5
            )
            + (PauliString(X.on(expected_qubits[0]) * 2.5))
            + (PauliString(Y.on(expected_qubits[1]) * 3.5))
        )

        # When
        paulisum = qubitop_to_paulisum(qubit_operator, qubits=expected_qubits)

        # Then
        self.assertEqual(paulisum.qubits, expected_qubits)
        self.assertEqual(paulisum, expected_paulisum)

    def test_get_ground_state_rdm_from_qubit_op(self):
        # Given
        n_sites = 2
        U = 5.0
        fhm = fermi_hubbard(
            x_dimension=n_sites,
            y_dimension=1,
            tunneling=1.0,
            coulomb=U,
            chemical_potential=U / 2,
            magnetic_field=0,
            periodic=False,
            spinless=False,
            particle_hole_symmetry=False,
        )
        fhm_qubit = jordan_wigner(fhm)
        fhm_int = get_interaction_operator(fhm)
        e, wf = jw_get_ground_state_at_particle_number(
            get_sparse_operator(fhm), n_sites
        )

        # When
        rdm = get_ground_state_rdm_from_qubit_op(
            qubit_operator=fhm_qubit, n_particles=n_sites
        )

        # Then
        self.assertAlmostEqual(e, rdm.expectation(fhm_int))

    def test_remove_inactive_orbitals(self):
        fermion_ham = load_interaction_operator(
            os.path.dirname(__file__) + "/../testing/hamiltonian_HeH_plus_STO-3G.json"
        )
        frozen_ham = remove_inactive_orbitals(fermion_ham, 1, 1)
        self.assertEqual(frozen_ham.one_body_tensor.shape[0], 2)

        hf_energy = hf_rdm(1, 1, 2).expectation(fermion_ham)
        self.assertAlmostEqual(frozen_ham.constant, hf_energy)
