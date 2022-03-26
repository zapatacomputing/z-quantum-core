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
"""Tests for operator_utils."""

import itertools
import os
import unittest

import numpy
import sympy
from scipy.sparse import csc_matrix
from zquantum.core.openfermion.config import DATA_DIRECTORY
from zquantum.core.openfermion.hamiltonians import fermi_hubbard
from zquantum.core.openfermion.ops.operators import (
    FermionOperator,
    IsingOperator,
    QubitOperator,
)
from zquantum.core.openfermion.ops.representations import InteractionOperator
from zquantum.core.openfermion.testing.testing_utils import random_interaction_operator
from zquantum.core.openfermion.transforms.opconversions import (
    bravyi_kitaev,
    jordan_wigner,
)
from zquantum.core.openfermion.transforms.repconversions import get_interaction_operator
from zquantum.core.openfermion.utils.operator_utils import (
    OperatorUtilsError,
    count_qubits,
    get_file_path,
    hermitian_conjugated,
    is_hermitian,
    is_identity,
    load_operator,
    save_operator,
)


class OperatorUtilsTest(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 5
        self.fermion_term = FermionOperator("1^ 2^ 3 4", -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term
        )
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.interaction_operator = get_interaction_operator(self.fermion_operator)
        self.ising_operator = IsingOperator("[Z0] + [Z1] + [Z2] + [Z3] + [Z4]")

    def test_n_qubits_single_fermion_term(self):
        self.assertEqual(self.n_qubits, count_qubits(self.fermion_term))

    def test_n_qubits_fermion_operator(self):
        self.assertEqual(self.n_qubits, count_qubits(self.fermion_operator))

    def test_n_qubits_qubit_operator(self):
        self.assertEqual(self.n_qubits, count_qubits(self.qubit_operator))

    def test_n_qubits_interaction_operator(self):
        self.assertEqual(self.n_qubits, count_qubits(self.interaction_operator))

    def test_n_qubits_ising_operator(self):
        self.assertEqual(self.n_qubits, count_qubits(self.ising_operator))

    def test_n_qubits_bad_type(self):
        with self.assertRaises(TypeError):
            count_qubits("twelve")

    def test_is_identity_unit_fermionoperator(self):
        self.assertTrue(is_identity(FermionOperator(())))

    def test_is_identity_double_of_unit_fermionoperator(self):
        self.assertTrue(is_identity(2.0 * FermionOperator(())))

    def test_is_identity_unit_qubitoperator(self):
        self.assertTrue(is_identity(QubitOperator(())))

    def test_is_identity_double_of_unit_qubitoperator(self):
        self.assertTrue(is_identity(QubitOperator((), 2.0)))

    def test_not_is_identity_single_term_fermionoperator(self):
        self.assertFalse(is_identity(FermionOperator("1^")))

    def test_not_is_identity_single_term_qubitoperator(self):
        self.assertFalse(is_identity(QubitOperator("X1")))

    def test_not_is_identity_zero_qubitoperator(self):
        self.assertFalse(is_identity(QubitOperator()))

    def test_is_identity_bad_type(self):
        with self.assertRaises(TypeError):
            _ = is_identity("eleven")


class HermitianConjugatedTest(unittest.TestCase):
    def test_hermitian_conjugated_qubit_op(self):
        """Test conjugating QubitOperators."""
        op = QubitOperator()
        op_hc = hermitian_conjugated(op)
        correct_op = op
        self.assertEqual(op_hc, correct_op)

        op = QubitOperator("X0 Y1", 2.0)
        op_hc = hermitian_conjugated(op)
        correct_op = op
        self.assertEqual(op_hc, correct_op)

        op = QubitOperator("X0 Y1", 2.0j)
        op_hc = hermitian_conjugated(op)
        correct_op = QubitOperator("X0 Y1", -2.0j)
        self.assertEqual(op_hc, correct_op)

        op = QubitOperator("X0 Y1", 2.0) + QubitOperator("Z4 X5 Y7", 3.0j)
        op_hc = hermitian_conjugated(op)
        correct_op = QubitOperator("X0 Y1", 2.0) + QubitOperator("Z4 X5 Y7", -3.0j)
        self.assertEqual(op_hc, correct_op)

    def test_hermitian_conjugated_qubit_op_consistency(self):
        """Some consistency checks for conjugating QubitOperators."""
        ferm_op = (
            FermionOperator("1^ 2")
            + FermionOperator("2 3 4")
            + FermionOperator("2^ 7 9 11^")
        )

        # Check that hermitian conjugation commutes with transforms
        self.assertEqual(
            jordan_wigner(hermitian_conjugated(ferm_op)),
            hermitian_conjugated(jordan_wigner(ferm_op)),
        )
        self.assertEqual(
            bravyi_kitaev(hermitian_conjugated(ferm_op)),
            hermitian_conjugated(bravyi_kitaev(ferm_op)),
        )

    def test_hermitian_conjugate_empty(self):
        op = FermionOperator()
        op = hermitian_conjugated(op)
        self.assertEqual(op, FermionOperator())

    def test_hermitian_conjugate_simple(self):
        op = FermionOperator("1^")
        op_hc = FermionOperator("1")
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

    def test_hermitian_conjugate_complex_const(self):
        op = FermionOperator("1^ 3", 3j)
        op_hc = -3j * FermionOperator("3^ 1")
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

    def test_hermitian_conjugate_notordered(self):
        op = FermionOperator("1 3^ 3 3^", 3j)
        op_hc = -3j * FermionOperator("3 3^ 3 1^")
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

    def test_hermitian_conjugate_semihermitian(self):
        op = (
            FermionOperator()
            + 2j * FermionOperator("1^ 3")
            + FermionOperator("3^ 1") * -2j
            + FermionOperator("2^ 2", 0.1j)
        )
        op_hc = (
            FermionOperator()
            + FermionOperator("1^ 3", 2j)
            + FermionOperator("3^ 1", -2j)
            + FermionOperator("2^ 2", -0.1j)
        )
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

    def test_hermitian_conjugated_empty(self):
        op = FermionOperator()
        self.assertEqual(op, hermitian_conjugated(op))

    def test_hermitian_conjugated_simple(self):
        op = FermionOperator("0")
        op_hc = FermionOperator("0^")
        self.assertEqual(op_hc, hermitian_conjugated(op))

    def test_hermitian_conjugated_complex_const(self):
        op = FermionOperator("2^ 2", 3j)
        op_hc = FermionOperator("2^ 2", -3j)
        self.assertEqual(op_hc, hermitian_conjugated(op))

    def test_hermitian_conjugated_multiterm(self):
        op = FermionOperator("1^ 2") + FermionOperator("2 3 4")
        op_hc = FermionOperator("2^ 1") + FermionOperator("4^ 3^ 2^")
        self.assertEqual(op_hc, hermitian_conjugated(op))

    def test_hermitian_conjugated_semihermitian(self):
        op = (
            FermionOperator()
            + 2j * FermionOperator("1^ 3")
            + FermionOperator("3^ 1") * -2j
            + FermionOperator("2^ 2", 0.1j)
        )
        op_hc = (
            FermionOperator()
            + FermionOperator("1^ 3", 2j)
            + FermionOperator("3^ 1", -2j)
            + FermionOperator("2^ 2", -0.1j)
        )
        self.assertEqual(op_hc, hermitian_conjugated(op))

    def test_hermitian_conjugated_interaction_operator(self):
        for n_orbitals, _ in itertools.product((1, 2, 5), range(5)):
            operator = random_interaction_operator(n_orbitals)
            qubit_operator = jordan_wigner(operator)
            conjugate_operator = hermitian_conjugated(operator)
            conjugate_qubit_operator = jordan_wigner(conjugate_operator)
            assert hermitian_conjugated(qubit_operator) == conjugate_qubit_operator

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            _ = is_hermitian("a")

        with self.assertRaises(TypeError):
            _ = hermitian_conjugated(1)


class IsHermitianTest(unittest.TestCase):
    def test_fermion_operator_zero(self):
        op = FermionOperator()
        self.assertTrue(is_hermitian(op))

    def test_fermion_operator_identity(self):
        op = FermionOperator(())
        self.assertTrue(is_hermitian(op))

    def test_fermion_operator_nonhermitian(self):
        op = FermionOperator("0^ 1 2^ 3")
        self.assertFalse(is_hermitian(op))

    def test_fermion_operator_hermitian(self):
        op = FermionOperator("0^ 1 2^ 3")
        op += FermionOperator("3^ 2 1^ 0")
        self.assertTrue(is_hermitian(op))

        op = fermi_hubbard(2, 2, 1.0, 1.0)
        self.assertTrue(is_hermitian(op))

        # TODO: insert bose_hubbard here
        # op = fermi_hubbard(2, 2, 1., 1.)
        # self.assertTrue(is_hermitian(op))

    def test_qubit_operator_zero(self):
        op = QubitOperator()
        self.assertTrue(is_hermitian(op))

    def test_qubit_operator_identity(self):
        op = QubitOperator(())
        self.assertTrue(is_hermitian(op))

    def test_qubit_operator_nonhermitian(self):
        op = QubitOperator("X0 Y2 Z5", 1.0 + 2.0j)
        self.assertFalse(is_hermitian(op))

    def test_qubit_operator_hermitian(self):
        op = QubitOperator("X0 Y2 Z5", 1.0 + 2.0j)
        op += QubitOperator("X0 Y2 Z5", 1.0 - 2.0j)
        self.assertTrue(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_zero(self):
        op = numpy.zeros((4, 4))
        self.assertTrue(is_hermitian(op))
        op = csc_matrix(op)
        self.assertTrue(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_identity(self):
        op = numpy.eye(4)
        self.assertTrue(is_hermitian(op))
        op = csc_matrix(op)
        self.assertTrue(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_nonhermitian(self):
        op = numpy.arange(16).reshape((4, 4))
        self.assertFalse(is_hermitian(op))
        op = csc_matrix(op)
        self.assertFalse(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_hermitian(self):
        op = numpy.arange(16, dtype=complex).reshape((4, 4))
        op += 1.0j * op
        op += op.T.conj()
        self.assertTrue(is_hermitian(op))
        op = csc_matrix(op)
        self.assertTrue(is_hermitian(op))

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            _ = is_hermitian("a")


class SaveLoadOperatorTest(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 5
        self.fermion_term = FermionOperator("1^ 2^ 3 4", -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term
        )
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.file_name = "test_file"

        self.bad_operator_filename = "bad_file.data"
        bad_op = "A:\nB"
        with open(os.path.join(DATA_DIRECTORY, self.bad_operator_filename), "w") as fid:
            fid.write(bad_op)

    def tearDown(self):
        file_path = os.path.join(DATA_DIRECTORY, self.file_name + ".data")
        if os.path.isfile(file_path):
            os.remove(file_path)
        file_path = os.path.join(DATA_DIRECTORY, self.bad_operator_filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    def test_save_sympy_plaintext(self):
        operator = FermionOperator("1^", sympy.Symbol("x"))
        with self.assertRaises(TypeError):
            save_operator(operator, self.file_name, plain_text=True)

    def test_raises_error_sympy(self):
        operator = FermionOperator("1^", sympy.Symbol("x"))
        with self.assertRaises(TypeError):
            save_operator(operator, self.file_name, plain_text=False)

    def test_save_and_load_fermion_operators(self):
        save_operator(self.fermion_operator, self.file_name)
        loaded_fermion_operator = load_operator(self.file_name)
        self.assertEqual(
            self.fermion_operator,
            loaded_fermion_operator,
            msg=str(self.fermion_operator - loaded_fermion_operator),
        )

    def test_save_and_load_fermion_operators_readably(self):
        save_operator(self.fermion_operator, self.file_name, plain_text=True)
        loaded_fermion_operator = load_operator(self.file_name, plain_text=True)
        self.assertTrue(self.fermion_operator == loaded_fermion_operator)

    def test_save_and_load_qubit_operators(self):
        save_operator(self.qubit_operator, self.file_name)
        loaded_qubit_operator = load_operator(self.file_name)
        self.assertTrue(self.qubit_operator == loaded_qubit_operator)

    def test_save_and_load_qubit_operators_readably(self):
        save_operator(self.qubit_operator, self.file_name, plain_text=True)
        loaded_qubit_operator = load_operator(self.file_name, plain_text=True)
        self.assertEqual(self.qubit_operator, loaded_qubit_operator)

    def test_load_bad_operator(self):
        with self.assertRaises(TypeError):
            load_operator(self.bad_operator_filename, plain_text=True)

    def test_save_readably(self):
        save_operator(self.fermion_operator, self.file_name, plain_text=True)
        file_path = os.path.join(DATA_DIRECTORY, self.file_name + ".data")
        with open(file_path, "r") as f:
            self.assertEqual(
                f.read(),
                "\n".join(
                    ["FermionOperator:", "-3.17 [1^ 2^ 3 4] +", "-3.17 [4^ 3^ 2 1]"]
                ),
            )

    def test_save_no_filename_operator_utils_error(self):
        with self.assertRaises(OperatorUtilsError):
            save_operator(self.fermion_operator)

    def test_basic_save(self):
        save_operator(self.fermion_operator, self.file_name)

    def test_save_interaction_operator_not_implemented(self):
        constant = 100.0
        one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
        two_body = numpy.zeros(
            (self.n_qubits, self.n_qubits, self.n_qubits, self.n_qubits), float
        )
        one_body[1, 1] = 10.0
        two_body[1, 2, 3, 4] = 12.0
        interaction_operator = InteractionOperator(constant, one_body, two_body)
        with self.assertRaises(NotImplementedError):
            save_operator(interaction_operator, self.file_name)

    def test_save_on_top_of_existing_operator_utils_error(self):
        save_operator(self.fermion_operator, self.file_name)
        with self.assertRaises(OperatorUtilsError):
            save_operator(self.fermion_operator, self.file_name)

    def test_save_on_top_of_existing_operator_error_with_explicit_flag(self):
        save_operator(self.fermion_operator, self.file_name)
        with self.assertRaises(OperatorUtilsError):
            save_operator(self.fermion_operator, self.file_name, allow_overwrite=False)

    def test_overwrite_flag_save_on_top_of_existing_operator(self):
        save_operator(self.fermion_operator, self.file_name)
        save_operator(self.fermion_operator, self.file_name, allow_overwrite=True)
        fermion_operator = load_operator(self.file_name)

        self.assertEqual(fermion_operator, self.fermion_operator)

    def test_load_bad_type(self):
        with self.assertRaises(TypeError):
            _ = load_operator("bad_type_operator")

    def test_save_bad_type(self):
        with self.assertRaises(TypeError):
            save_operator("ping", "somewhere")


class GetFileDirTest(unittest.TestCase):
    def setUp(self):
        self.filename = "foo"
        self.datadirname = "data"

    def test_file_load(self):
        """Test if file name is acquired correctly"""
        filepath = get_file_path(self.filename, None)
        self.assertEqual(filepath, DATA_DIRECTORY + "/" + self.filename + ".data")

        filepath = get_file_path(self.filename, self.datadirname)
        self.assertEqual(filepath, self.datadirname + "/" + self.filename + ".data")
