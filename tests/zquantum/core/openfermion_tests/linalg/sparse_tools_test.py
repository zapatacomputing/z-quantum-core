#   Copyright 2017 The OpenFermion Developers
#   Modifications copyright 2022 Zapata Computing, Inc. for compatibility reasons.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests for sparse_tools.py."""
import os
import unittest

import numpy
import scipy
from scipy.linalg import eigh, norm
from scipy.sparse import csc_matrix
from scipy.special import comb
from zquantum.core.openfermion.chem import MolecularData
from zquantum.core.openfermion.config import DATA_DIRECTORY
from zquantum.core.openfermion.hamiltonians import fermi_hubbard
from zquantum.core.openfermion.hamiltonians.special_operators import number_operator
from zquantum.core.openfermion.linalg.sparse_tools import (
    _iterate_basis_,
    eigenspectrum,
    expectation,
    get_ground_state,
    get_number_preserving_sparse_operator,
    get_sparse_operator,
    inner_product,
    jordan_wigner_sparse,
    jw_configuration_state,
    jw_get_ground_state_at_particle_number,
    jw_hartree_fock_state,
    jw_number_indices,
    jw_number_restrict_operator,
    qubit_operator_sparse,
    sparse_eigenspectrum,
)
from zquantum.core.openfermion.ops.operators import FermionOperator, QubitOperator
from zquantum.core.openfermion.transforms.opconversions import (
    get_fermion_operator,
    jordan_wigner,
)
from zquantum.core.openfermion.transforms.repconversions import get_interaction_operator
from zquantum.core.openfermion.utils.operator_utils import (
    count_qubits,
    hermitian_conjugated,
    is_hermitian,
)


class EigenSpectrumTest(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 5
        self.fermion_term = FermionOperator("1^ 2^ 3 4", -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term
        )
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.interaction_operator = get_interaction_operator(self.fermion_operator)

    def test_eigenspectrum(self):
        fermion_eigenspectrum = eigenspectrum(self.fermion_operator)
        qubit_eigenspectrum = eigenspectrum(self.qubit_operator)
        interaction_eigenspectrum = eigenspectrum(self.interaction_operator)
        for i in range(2 ** self.n_qubits):
            self.assertAlmostEqual(fermion_eigenspectrum[i], qubit_eigenspectrum[i])
            self.assertAlmostEqual(
                fermion_eigenspectrum[i], interaction_eigenspectrum[i]
            )


class SparseOperatorTest(unittest.TestCase):
    def test_qubit_jw_fermion_integration(self):

        # Initialize a random fermionic operator.
        fermion_operator = FermionOperator(((3, 1), (2, 1), (1, 0), (0, 0)), -4.3)
        fermion_operator += FermionOperator(((3, 1), (1, 0)), 8.17)
        fermion_operator += 3.2 * FermionOperator()

        # Map to qubits and compare matrix versions.
        qubit_operator = jordan_wigner(fermion_operator)
        qubit_sparse = get_sparse_operator(qubit_operator)
        qubit_spectrum = sparse_eigenspectrum(qubit_sparse)
        fermion_sparse = jordan_wigner_sparse(fermion_operator)
        fermion_spectrum = sparse_eigenspectrum(fermion_sparse)
        self.assertAlmostEqual(
            0.0, numpy.amax(numpy.absolute(fermion_spectrum - qubit_spectrum))
        )


class JordanWignerSparseTest(unittest.TestCase):
    def test_jw_sparse_0create(self):
        expected = csc_matrix(([1], ([1], [0])), shape=(2, 2))
        self.assertTrue(
            numpy.allclose(jordan_wigner_sparse(FermionOperator("0^")).A, expected.A)
        )

    def test_jw_sparse_1annihilate(self):
        expected = csc_matrix(([1, -1], ([0, 2], [1, 3])), shape=(4, 4))
        self.assertTrue(
            numpy.allclose(jordan_wigner_sparse(FermionOperator("1")).A, expected.A)
        )

    def test_jw_sparse_0create_2annihilate(self):
        expected = csc_matrix(
            ([-1j, 1j], ([4, 6], [1, 3])), shape=(8, 8), dtype=numpy.complex128
        )
        self.assertTrue(
            numpy.allclose(
                jordan_wigner_sparse(FermionOperator("0^ 2", -1j)).A, expected.A
            )
        )

    def test_jw_sparse_0create_3annihilate(self):
        expected = csc_matrix(
            ([-1j, 1j, 1j, -1j], ([8, 10, 12, 14], [1, 3, 5, 7])),
            shape=(16, 16),
            dtype=numpy.complex128,
        )
        self.assertTrue(
            numpy.allclose(
                jordan_wigner_sparse(FermionOperator("0^ 3", -1j)).A, expected.A
            )
        )

    def test_jw_sparse_twobody(self):
        expected = csc_matrix(([1, 1], ([6, 14], [5, 13])), shape=(16, 16))
        self.assertTrue(
            numpy.allclose(
                jordan_wigner_sparse(FermionOperator("2^ 1^ 1 3")).A, expected.A
            )
        )

    def test_qubit_operator_sparse_n_qubits_too_small(self):
        with self.assertRaises(ValueError):
            qubit_operator_sparse(QubitOperator("X3"), 1)

    def test_qubit_operator_sparse_n_qubits_not_specified(self):
        expected = csc_matrix(
            ([1, 1, 1, 1], ([1, 0, 3, 2], [0, 1, 2, 3])), shape=(4, 4)
        )
        self.assertTrue(
            numpy.allclose(qubit_operator_sparse(QubitOperator("X1")).A, expected.A)
        )


class ComputationalBasisStateTest(unittest.TestCase):
    def test_computational_basis_state(self):
        comp_basis_state = jw_configuration_state([0, 2, 5], 7)
        self.assertAlmostEqual(comp_basis_state[82], 1.0)
        self.assertAlmostEqual(sum(comp_basis_state), 1.0)


class JWHartreeFockStateTest(unittest.TestCase):
    def test_jw_hartree_fock_state(self):
        hartree_fock_state = jw_hartree_fock_state(3, 7)
        self.assertAlmostEqual(hartree_fock_state[112], 1.0)
        self.assertAlmostEqual(sum(hartree_fock_state), 1.0)


class JWNumberIndicesTest(unittest.TestCase):
    def test_jw_sparse_index(self):
        """Test the indexing scheme for selecting specific particle numbers"""
        expected = [1, 2]
        calculated_indices = jw_number_indices(1, 2)
        self.assertEqual(expected, calculated_indices)

        expected = [3]
        calculated_indices = jw_number_indices(2, 2)
        self.assertEqual(expected, calculated_indices)

    def test_jw_number_indices(self):
        n_qubits = numpy.random.randint(1, 12)
        n_particles = numpy.random.randint(n_qubits + 1)

        number_indices = jw_number_indices(n_particles, n_qubits)
        subspace_dimension = len(number_indices)

        self.assertEqual(subspace_dimension, comb(n_qubits, n_particles))

        for index in number_indices:
            binary_string = bin(index)[2:].zfill(n_qubits)
            n_ones = binary_string.count("1")
            self.assertEqual(n_ones, n_particles)


class JWNumberRestrictOperatorTest(unittest.TestCase):
    def test_jw_restrict_operator(self):
        """Test the scheme for restricting JW encoded operators to number"""
        # Make a Hamiltonian that cares mostly about number of electrons
        n_qubits = 4
        target_electrons = 2
        penalty_const = 10.0
        number_sparse = jordan_wigner_sparse(number_operator(n_qubits))
        bias_sparse = jordan_wigner_sparse(
            sum(
                [FermionOperator(((i, 1), (i, 0)), 1.0) for i in range(n_qubits)],
                FermionOperator(),
            )
        )
        hamiltonian_sparse = (
            penalty_const
            * (
                number_sparse - target_electrons * scipy.sparse.identity(2 ** n_qubits)
            ).dot(
                number_sparse - target_electrons * scipy.sparse.identity(2 ** n_qubits)
            )
            + bias_sparse
        )

        restricted_hamiltonian = jw_number_restrict_operator(
            hamiltonian_sparse, target_electrons, n_qubits
        )
        true_eigvals, _ = eigh(hamiltonian_sparse.A)
        test_eigvals, _ = eigh(restricted_hamiltonian.A)

        self.assertAlmostEqual(norm(true_eigvals[:6] - test_eigvals[:6]), 0.0)

    def test_jw_restrict_operator_hopping_to_1_particle(self):
        hop = FermionOperator("3^ 1") + FermionOperator("1^ 3")
        hop_sparse = jordan_wigner_sparse(hop, n_qubits=4)
        hop_restrict = jw_number_restrict_operator(hop_sparse, 1, n_qubits=4)
        expected = csc_matrix(([1, 1], ([0, 2], [2, 0])), shape=(4, 4))

        self.assertTrue(numpy.allclose(hop_restrict.A, expected.A))

    def test_jw_restrict_operator_interaction_to_1_particle(self):
        interaction = FermionOperator("3^ 2^ 4 1")
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        interaction_restrict = jw_number_restrict_operator(
            interaction_sparse, 1, n_qubits=6
        )
        expected = csc_matrix(([], ([], [])), shape=(6, 6))

        self.assertTrue(numpy.allclose(interaction_restrict.A, expected.A))

    def test_jw_restrict_operator_interaction_to_2_particles(self):
        interaction = FermionOperator("3^ 2^ 4 1") + FermionOperator("4^ 1^ 3 2")
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        interaction_restrict = jw_number_restrict_operator(
            interaction_sparse, 2, n_qubits=6
        )

        dim = 6 * 5 // 2  # shape of new sparse array

        # 3^ 2^ 4 1 maps 2**4 + 2 = 18 to 2**3 + 2**2 = 12 and vice versa;
        # in the 2-particle subspace (1, 4) and (2, 3) are 7th and 9th.
        expected = csc_matrix(([-1, -1], ([7, 9], [9, 7])), shape=(dim, dim))

        self.assertTrue(numpy.allclose(interaction_restrict.A, expected.A))

    def test_jw_restrict_operator_hopping_to_1_particle_default_nqubits(self):
        interaction = FermionOperator("3^ 2^ 4 1") + FermionOperator("4^ 1^ 3 2")
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        # n_qubits should default to 6
        interaction_restrict = jw_number_restrict_operator(interaction_sparse, 2)

        dim = 6 * 5 // 2  # shape of new sparse array

        # 3^ 2^ 4 1 maps 2**4 + 2 = 18 to 2**3 + 2**2 = 12 and vice versa;
        # in the 2-particle subspace (1, 4) and (2, 3) are 7th and 9th.
        expected = csc_matrix(([-1, -1], ([7, 9], [9, 7])), shape=(dim, dim))

        self.assertTrue(numpy.allclose(interaction_restrict.A, expected.A))


class JWGetGroundStatesByParticleNumberTest(unittest.TestCase):
    def test_jw_get_ground_state_at_particle_number_herm_conserving(self):
        # Initialize a particle-number-conserving Hermitian operator
        ferm_op = (
            FermionOperator("0^ 1")
            + FermionOperator("1^ 0")
            + FermionOperator("1^ 2")
            + FermionOperator("2^ 1")
            + FermionOperator("1^ 3", -0.4)
            + FermionOperator("3^ 1", -0.4)
        )
        jw_hamiltonian = jordan_wigner(ferm_op)
        sparse_operator = get_sparse_operator(jw_hamiltonian)
        n_qubits = 4

        num_op = get_sparse_operator(number_operator(n_qubits))

        # Test each possible particle number
        for particle_number in range(n_qubits):
            # Get the ground energy and ground state at this particle number
            energy, state = jw_get_ground_state_at_particle_number(
                sparse_operator, particle_number
            )

            # Check that it's an eigenvector with the correct eigenvalue
            self.assertTrue(numpy.allclose(sparse_operator.dot(state), energy * state))

            # Check that it has the correct particle number
            num = expectation(num_op, state)
            self.assertAlmostEqual(num, particle_number)

    def test_jw_get_ground_state_at_particle_number_hubbard(self):

        model = fermi_hubbard(2, 2, 1.0, 4.0)
        sparse_operator = get_sparse_operator(model)
        n_qubits = count_qubits(model)
        num_op = get_sparse_operator(number_operator(n_qubits))

        # Test each possible particle number
        for particle_number in range(n_qubits):
            # Get the ground energy and ground state at this particle number
            energy, state = jw_get_ground_state_at_particle_number(
                sparse_operator, particle_number
            )

            # Check that it's an eigenvector with the correct eigenvalue
            self.assertTrue(numpy.allclose(sparse_operator.dot(state), energy * state))

            # Check that it has the correct particle number
            num = expectation(num_op, state)
            self.assertAlmostEqual(num, particle_number)


class GroundStateTest(unittest.TestCase):
    def test_get_ground_state_hermitian(self):
        ground = get_ground_state(
            get_sparse_operator(QubitOperator("Y0 X1") + QubitOperator("Z0 Z1"))
        )
        expected_state = csc_matrix(
            ([1j, 1], ([1, 2], [0, 0])), shape=(4, 1), dtype=numpy.complex128
        ).A
        expected_state /= numpy.sqrt(2.0)

        self.assertAlmostEqual(ground[0], -2)
        self.assertAlmostEqual(
            numpy.absolute(expected_state.T.conj().dot(ground[1]))[0], 1.0
        )


class ExpectationTest(unittest.TestCase):
    def test_expectation_correct_sparse_matrix(self):
        operator = get_sparse_operator(QubitOperator("X0"), n_qubits=2)
        vector = numpy.array([0.0, 1.0j, 0.0, 1.0j])
        self.assertAlmostEqual(expectation(operator, vector), 2.0)

        density_matrix = scipy.sparse.csc_matrix(
            numpy.outer(vector, numpy.conjugate(vector))
        )
        self.assertAlmostEqual(expectation(operator, density_matrix), 2.0)

    def test_expectation_handles_column_vector(self):
        operator = get_sparse_operator(QubitOperator("X0"), n_qubits=2)
        vector = numpy.array([[0.0], [1.0j], [0.0], [1.0j]])
        self.assertAlmostEqual(expectation(operator, vector), 2.0)

    def test_expectation_correct_zero(self):
        operator = get_sparse_operator(QubitOperator("X0"), n_qubits=2)
        vector = numpy.array([1j, -1j, -1j, -1j])
        self.assertAlmostEqual(expectation(operator, vector), 0.0)


class InnerProductTest(unittest.TestCase):
    def test_inner_product(self):
        state_1 = numpy.array([1.0, 1.0j])
        state_2 = numpy.array([1.0, -1.0j])

        self.assertAlmostEqual(inner_product(state_1, state_1), 2.0)
        self.assertAlmostEqual(inner_product(state_1, state_2), 0.0)


class GetNumberPreservingSparseOperatorIntegrationTestLiH(unittest.TestCase):
    def setUp(self):
        # Set up molecule.
        geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.45))]
        basis = "sto-3g"
        multiplicity = 1
        filename = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45")
        self.molecule = MolecularData(geometry, basis, multiplicity, filename=filename)
        self.molecule.load()

        # Get molecular Hamiltonian.
        self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()

        self.hubbard_hamiltonian = fermi_hubbard(
            2, 2, 1.0, 4.0, chemical_potential=0.2, magnetic_field=0.0, spinless=False
        )

    def test_exceptions(self):
        op = FermionOperator("1")
        with self.assertRaises(ValueError):
            _ = get_number_preserving_sparse_operator(op, 2, 1)

    def test_number_on_reference(self):
        sum_n_op = FermionOperator()
        sum_sparse_n_op = get_number_preserving_sparse_operator(
            sum_n_op,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False,
        )

        space_size = sum_sparse_n_op.shape[0]
        reference = numpy.zeros((space_size))
        reference[0] = 1.0

        for i in range(self.molecule.n_qubits):
            n_op = FermionOperator(((i, 1), (i, 0)))
            sum_n_op += n_op

            sparse_n_op = get_number_preserving_sparse_operator(
                n_op,
                self.molecule.n_qubits,
                self.molecule.n_electrons,
                spin_preserving=False,
            )

            sum_sparse_n_op += sparse_n_op

            expectation = reference.dot(sparse_n_op.dot(reference))

            if i < self.molecule.n_electrons:
                assert expectation == 1.0
            else:
                assert expectation == 0.0

        convert_after_adding = get_number_preserving_sparse_operator(
            sum_n_op,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False,
        )

        assert scipy.sparse.linalg.norm(convert_after_adding - sum_sparse_n_op) < 1e-9

        assert (
            reference.dot(sum_sparse_n_op.dot(reference)) - self.molecule.n_electrons
            < 1e-9
        )

    def test_space_size_correct(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
        )

        space_size = sparse_ham.shape[0]

        # Naive Hilbert space size is 2**12, or 4096.
        assert space_size == 225

    def test_hf_energy(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
        )

        space_size = sparse_ham.shape[0]
        reference = numpy.zeros((space_size))
        reference[0] = 1.0

        sparse_hf_energy = reference.dot(sparse_ham.dot(reference))

        assert numpy.linalg.norm(sparse_hf_energy - self.molecule.hf_energy) < 1e-9

    def test_one_body_hf_energy(self):
        one_body_part = self.molecular_hamiltonian
        one_body_part.two_body_tensor = numpy.zeros_like(one_body_part.two_body_tensor)

        one_body_fop = get_fermion_operator(one_body_part)
        one_body_regular_sparse_op = get_sparse_operator(one_body_fop)

        make_hf_fop = FermionOperator(((3, 1), (2, 1), (1, 1), (0, 1)))
        make_hf_sparse_op = get_sparse_operator(make_hf_fop, n_qubits=12)

        hf_state = numpy.zeros((2 ** 12))
        hf_state[0] = 1.0
        hf_state = make_hf_sparse_op.dot(hf_state)

        regular_sparse_hf_energy = (
            hf_state.dot(one_body_regular_sparse_op.dot(hf_state))
        ).real

        one_body_sparse_op = get_number_preserving_sparse_operator(
            one_body_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
        )

        space_size = one_body_sparse_op.shape[0]
        reference = numpy.zeros((space_size))
        reference[0] = 1.0

        sparse_hf_energy = reference.dot(one_body_sparse_op.dot(reference))

        assert numpy.linalg.norm(sparse_hf_energy - regular_sparse_hf_energy) < 1e-9

    def test_ground_state_energy(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
        )

        eig_val, _ = scipy.sparse.linalg.eigsh(sparse_ham, k=1, which="SA")

        assert numpy.abs(eig_val[0] - self.molecule.fci_energy) < 1e-9

    def test_doubles_are_subset(self):
        reference_determinants = [
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
            ],
        ]

        for reference_determinant in reference_determinants:
            reference_determinant = numpy.asarray(reference_determinant)
            doubles_state_array = numpy.asarray(
                list(
                    _iterate_basis_(
                        reference_determinant, excitation_level=2, spin_preserving=True
                    )
                )
            )
            doubles_int_state_array = doubles_state_array.dot(
                1 << numpy.arange(doubles_state_array.shape[1])[::-1]
            )

            all_state_array = numpy.asarray(
                list(
                    _iterate_basis_(
                        reference_determinant, excitation_level=4, spin_preserving=True
                    )
                )
            )
            all_int_state_array = all_state_array.dot(
                1 << numpy.arange(all_state_array.shape[1])[::-1]
            )

            for item in doubles_int_state_array:
                assert item in all_int_state_array

        for reference_determinant in reference_determinants:
            reference_determinant = numpy.asarray(reference_determinant)
            doubles_state_array = numpy.asarray(
                list(
                    _iterate_basis_(
                        reference_determinant, excitation_level=2, spin_preserving=True
                    )
                )
            )
            doubles_int_state_array = doubles_state_array.dot(
                1 << numpy.arange(doubles_state_array.shape[1])[::-1]
            )

            all_state_array = numpy.asarray(
                list(
                    _iterate_basis_(
                        reference_determinant, excitation_level=4, spin_preserving=False
                    )
                )
            )
            all_int_state_array = all_state_array.dot(
                1 << numpy.arange(all_state_array.shape[1])[::-1]
            )

            for item in doubles_int_state_array:
                assert item in all_int_state_array

        for reference_determinant in reference_determinants:
            reference_determinant = numpy.asarray(reference_determinant)
            doubles_state_array = numpy.asarray(
                list(
                    _iterate_basis_(
                        reference_determinant, excitation_level=2, spin_preserving=False
                    )
                )
            )
            doubles_int_state_array = doubles_state_array.dot(
                1 << numpy.arange(doubles_state_array.shape[1])[::-1]
            )

            all_state_array = numpy.asarray(
                list(
                    _iterate_basis_(
                        reference_determinant, excitation_level=4, spin_preserving=False
                    )
                )
            )
            all_int_state_array = all_state_array.dot(
                1 << numpy.arange(all_state_array.shape[1])[::-1]
            )

            for item in doubles_int_state_array:
                assert item in all_int_state_array

    def test_full_ham_hermitian(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
        )

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1e-9

    def test_full_ham_hermitian_non_spin_preserving(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False,
        )

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1e-9

    def test_singles_simple_one_body_term_hermitian(self):
        fop = FermionOperator(((3, 1), (1, 0)))
        fop_conj = FermionOperator(((1, 1), (3, 0)))

        sparse_op = get_number_preserving_sparse_operator(
            fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1,
        )

        sparse_op_conj = get_number_preserving_sparse_operator(
            fop_conj,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1,
        )

        assert scipy.sparse.linalg.norm(sparse_op - sparse_op_conj.getH()) < 1e-9

    def test_singles_simple_two_body_term_hermitian(self):
        fop = FermionOperator(((3, 1), (8, 1), (1, 0), (4, 0)))
        fop_conj = FermionOperator(((4, 1), (1, 1), (8, 0), (3, 0)))

        sparse_op = get_number_preserving_sparse_operator(
            fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1,
        )

        sparse_op_conj = get_number_preserving_sparse_operator(
            fop_conj,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1,
        )

        assert scipy.sparse.linalg.norm(sparse_op - sparse_op_conj.getH()) < 1e-9

    def test_singles_repeating_two_body_term_hermitian(self):
        fop = FermionOperator(((3, 1), (1, 1), (5, 0), (1, 0)))
        fop_conj = FermionOperator(((5, 1), (1, 1), (3, 0), (1, 0)))

        sparse_op = get_number_preserving_sparse_operator(
            fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1,
        )

        sparse_op_conj = get_number_preserving_sparse_operator(
            fop_conj,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1,
        )

        assert scipy.sparse.linalg.norm(sparse_op - sparse_op_conj.getH()) < 1e-9

    def test_singles_ham_hermitian(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1,
        )

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1e-9

    def test_singles_ham_hermitian_non_spin_preserving(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False,
            excitation_level=1,
        )

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1e-9

    def test_cisd_energy(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=2,
        )

        eig_val, _ = scipy.sparse.linalg.eigsh(sparse_ham, k=1, which="SA")

        assert numpy.abs(eig_val[0] - self.molecule.cisd_energy) < 1e-9

    def test_cisd_energy_non_spin_preserving(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False,
            excitation_level=2,
        )

        eig_val, _ = scipy.sparse.linalg.eigsh(sparse_ham, k=1, which="SA")

        assert numpy.abs(eig_val[0] - self.molecule.cisd_energy) < 1e-9

    def test_cisd_matches_fci_energy_two_electron_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_cisd = get_number_preserving_sparse_operator(
            hamiltonian_fop, 8, 2, spin_preserving=True, excitation_level=2
        )

        sparse_ham_fci = get_sparse_operator(hamiltonian_fop, n_qubits=8)

        eig_val_cisd, _ = scipy.sparse.linalg.eigsh(sparse_ham_cisd, k=1, which="SA")
        eig_val_fci, _ = scipy.sparse.linalg.eigsh(sparse_ham_fci, k=1, which="SA")

        assert numpy.abs(eig_val_cisd[0] - eig_val_fci[0]) < 1e-9

    def test_weird_determinant_matches_fci_energy_two_electron_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_cisd = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            8,
            2,
            spin_preserving=True,
            excitation_level=2,
            reference_determinant=numpy.asarray(
                [False, False, True, True, False, False, False, False]
            ),
        )

        sparse_ham_fci = get_sparse_operator(hamiltonian_fop, n_qubits=8)

        eig_val_cisd, _ = scipy.sparse.linalg.eigsh(sparse_ham_cisd, k=1, which="SA")
        eig_val_fci, _ = scipy.sparse.linalg.eigsh(sparse_ham_fci, k=1, which="SA")

        assert numpy.abs(eig_val_cisd[0] - eig_val_fci[0]) < 1e-9

    def test_number_restricted_spectra_match_molecule(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham_number_preserving = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False,
        )

        sparse_ham = get_sparse_operator(hamiltonian_fop, self.molecule.n_qubits)

        sparse_ham_restricted_number_preserving = jw_number_restrict_operator(
            sparse_ham,
            n_electrons=self.molecule.n_electrons,
            n_qubits=self.molecule.n_qubits,
        )

        spectrum_from_new_sparse_method = sparse_eigenspectrum(
            sparse_ham_number_preserving
        )

        spectrum_from_old_sparse_method = sparse_eigenspectrum(
            sparse_ham_restricted_number_preserving
        )

        spectral_deviation = numpy.amax(
            numpy.absolute(
                spectrum_from_new_sparse_method - spectrum_from_old_sparse_method
            )
        )
        self.assertAlmostEqual(spectral_deviation, 0.0)

    def test_number_restricted_spectra_match_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_number_preserving = get_number_preserving_sparse_operator(
            hamiltonian_fop, 8, 4, spin_preserving=False
        )

        sparse_ham = get_sparse_operator(hamiltonian_fop, 8)

        sparse_ham_restricted_number_preserving = jw_number_restrict_operator(
            sparse_ham, n_electrons=4, n_qubits=8
        )

        spectrum_from_new_sparse_method = sparse_eigenspectrum(
            sparse_ham_number_preserving
        )

        spectrum_from_old_sparse_method = sparse_eigenspectrum(
            sparse_ham_restricted_number_preserving
        )

        spectral_deviation = numpy.amax(
            numpy.absolute(
                spectrum_from_new_sparse_method - spectrum_from_old_sparse_method
            )
        )
        self.assertAlmostEqual(spectral_deviation, 0.0)


class GetSparseOperatorQubitTest(unittest.TestCase):
    def test_sparse_matrix_Y(self):
        term = QubitOperator(((0, "Y"),))
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.data), [1j, -1j])
        self.assertEqual(list(sparse_operator.indices), [1, 0])
        self.assertTrue(is_hermitian(sparse_operator))

    def test_sparse_matrix_ZX(self):
        coefficient = 2.0
        operators = ((0, "Z"), (1, "X"))
        term = QubitOperator(operators, coefficient)
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.data), [2.0, 2.0, -2.0, -2.0])
        self.assertEqual(list(sparse_operator.indices), [1, 0, 3, 2])
        self.assertTrue(is_hermitian(sparse_operator))

    def test_sparse_matrix_ZIZ(self):
        operators = ((0, "Z"), (2, "Z"))
        term = QubitOperator(operators)
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.data), [1, -1, 1, -1, -1, 1, -1, 1])
        self.assertEqual(list(sparse_operator.indices), list(range(8)))
        self.assertTrue(is_hermitian(sparse_operator))

    def test_sparse_matrix_combo(self):
        qop = QubitOperator(((0, "Y"), (1, "X")), -0.1j) + QubitOperator(
            ((0, "X"), (1, "Z")), 3.0 + 2.0j
        )
        sparse_operator = get_sparse_operator(qop)

        self.assertEqual(
            list(sparse_operator.data),
            [3 + 2j, 0.1, 0.1, -3 - 2j, 3 + 2j, -0.1, -0.1, -3 - 2j],
        )
        self.assertEqual(list(sparse_operator.indices), [2, 3, 2, 3, 0, 1, 0, 1])

    def test_sparse_matrix_zero_1qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator((), 0.0), 1)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.data)), 0)
        self.assertEqual(sparse_operator.shape, (2, 2))

    def test_sparse_matrix_zero_5qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator((), 0.0), 5)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.data)), 0)
        self.assertEqual(sparse_operator.shape, (32, 32))

    def test_sparse_matrix_identity_1qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator(()), 1)
        self.assertEqual(list(sparse_operator.data), [1] * 2)
        self.assertEqual(sparse_operator.shape, (2, 2))

    def test_sparse_matrix_identity_5qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator(()), 5)
        self.assertEqual(list(sparse_operator.data), [1] * 32)
        self.assertEqual(sparse_operator.shape, (32, 32))

    def test_sparse_matrix_linearity(self):
        identity = QubitOperator(())
        zzzz = QubitOperator(tuple((i, "Z") for i in range(4)), 1.0)

        sparse1 = get_sparse_operator(identity + zzzz)
        sparse2 = get_sparse_operator(identity, 4) + get_sparse_operator(zzzz)

        self.assertEqual(list(sparse1.data), [2] * 8)
        self.assertEqual(list(sparse1.indices), [0, 3, 5, 6, 9, 10, 12, 15])
        self.assertEqual(list(sparse2.data), [2] * 8)
        self.assertEqual(list(sparse2.indices), [0, 3, 5, 6, 9, 10, 12, 15])


class GetSparseOperatorFermionTest(unittest.TestCase):
    def test_sparse_matrix_zero_n_qubit(self):
        sparse_operator = get_sparse_operator(FermionOperator.zero(), 4)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.data)), 0)
        self.assertEqual(sparse_operator.shape, (16, 16))
