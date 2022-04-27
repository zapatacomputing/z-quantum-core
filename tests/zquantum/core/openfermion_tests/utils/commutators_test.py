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
"""Tests for commutators.py."""
import unittest

from zquantum.core.openfermion.ops.operators import FermionOperator, QubitOperator
from zquantum.core.openfermion.transforms import jordan_wigner
from zquantum.core.openfermion.transforms.opconversions import normal_ordered
from zquantum.core.openfermion.utils import hermitian_conjugated
from zquantum.core.openfermion.utils.commutators import (
    anticommutator,
    commutator,
    double_commutator,
)


class CommutatorTest(unittest.TestCase):
    def setUp(self):
        self.fermion_term = FermionOperator("1^ 2^ 3 4", -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term
        )
        self.qubit_operator = jordan_wigner(self.fermion_operator)

    def test_commutes_identity(self):
        com = commutator(FermionOperator.identity(), FermionOperator("2^ 3", 2.3))
        self.assertEqual(com, FermionOperator.zero())

    def test_commutes_no_intersection(self):
        com = commutator(FermionOperator("2^ 3"), FermionOperator("4^ 5^ 3"))
        com = normal_ordered(com)
        self.assertEqual(com, FermionOperator.zero())

    def test_commutes_number_operators(self):
        com = commutator(FermionOperator("4^ 3^ 4 3"), FermionOperator("2^ 2"))
        com = normal_ordered(com)
        self.assertEqual(com, FermionOperator.zero())

    def test_commutator_hopping_operators(self):
        com = commutator(3 * FermionOperator("1^ 2"), FermionOperator("2^ 3"))
        com = normal_ordered(com)
        self.assertEqual(com, FermionOperator("1^ 3", 3))

    def test_commutator_hopping_with_single_number(self):
        com = commutator(FermionOperator("1^ 2", 1j), FermionOperator("1^ 1"))
        com = normal_ordered(com)
        self.assertEqual(com, -FermionOperator("1^ 2") * 1j)

    def test_commutator_hopping_with_double_number_one_intersection(self):
        com = commutator(FermionOperator("1^ 3"), FermionOperator("3^ 2^ 3 2"))
        com = normal_ordered(com)
        self.assertEqual(com, -FermionOperator("2^ 1^ 3 2"))

    def test_commutator_hopping_with_double_number_two_intersections(self):
        com = commutator(FermionOperator("2^ 3"), FermionOperator("3^ 2^ 3 2"))
        com = normal_ordered(com)
        self.assertEqual(com, FermionOperator.zero())

    def test_commutator(self):
        operator_a = FermionOperator("")
        self.assertEqual(
            FermionOperator.zero(), commutator(operator_a, self.fermion_operator)
        )
        operator_b = QubitOperator("X1 Y2")
        self.assertEqual(
            commutator(self.qubit_operator, operator_b),
            (self.qubit_operator * operator_b - operator_b * self.qubit_operator),
        )

    def test_commutator_operator_a_bad_type(self):
        with self.assertRaises(TypeError):
            commutator(1, self.fermion_operator)

    def test_commutator_operator_b_bad_type(self):
        with self.assertRaises(TypeError):
            commutator(self.qubit_operator, "hello")

    def test_commutator_not_same_type(self):
        with self.assertRaises(TypeError):
            commutator(self.fermion_operator, self.qubit_operator)


class AnticommutatorTest(unittest.TestCase):
    def test_canonical_anticommutation_relations(self):
        op_1 = FermionOperator("3")
        op_1_dag = FermionOperator("3^")
        op_2 = FermionOperator("4")
        op_2_dag = FermionOperator("4^")
        zero = FermionOperator.zero()
        one = FermionOperator.identity()

        self.assertEqual(one, normal_ordered(anticommutator(op_1, op_1_dag)))
        self.assertEqual(zero, normal_ordered(anticommutator(op_1, op_2)))
        self.assertEqual(zero, normal_ordered(anticommutator(op_1, op_2_dag)))
        self.assertEqual(zero, normal_ordered(anticommutator(op_1_dag, op_2)))
        self.assertEqual(zero, normal_ordered(anticommutator(op_1_dag, op_2_dag)))
        self.assertEqual(one, normal_ordered(anticommutator(op_2, op_2_dag)))

    def test_anticommutator_not_same_type(self):
        with self.assertRaises(TypeError):
            anticommutator(FermionOperator(), QubitOperator())


class DoubleCommutatorTest(unittest.TestCase):
    def test_double_commutator_no_intersection_with_union_of_second_two(self):
        com = double_commutator(
            FermionOperator("4^ 3^ 6 5"),
            FermionOperator("2^ 1 0"),
            FermionOperator("0^"),
        )
        self.assertEqual(com, FermionOperator.zero())

    def test_double_commutator_more_info_not_hopping(self):
        com = double_commutator(
            FermionOperator("3^ 2"),
            FermionOperator("2^ 3") + FermionOperator("3^ 2"),
            FermionOperator("4^ 2^ 4 2"),
            indices2=set([2, 3]),
            indices3=set([2, 4]),
            is_hopping_operator2=True,
            is_hopping_operator3=False,
        )
        self.assertEqual(
            com, (FermionOperator("4^ 2^ 4 2") - FermionOperator("4^ 3^ 4 3"))
        )

    def test_double_commtator_more_info_both_hopping(self):
        com = double_commutator(
            FermionOperator("4^ 3^ 4 3"),
            FermionOperator("1^ 2", 2.1) + FermionOperator("2^ 1", 2.1),
            FermionOperator("1^ 3", -1.3) + FermionOperator("3^ 1", -1.3),
            indices2=set([1, 2]),
            indices3=set([1, 3]),
            is_hopping_operator2=True,
            is_hopping_operator3=True,
        )
        self.assertEqual(
            com,
            (FermionOperator("4^ 3^ 4 2", 2.73) + FermionOperator("4^ 2^ 4 3", 2.73)),
        )
