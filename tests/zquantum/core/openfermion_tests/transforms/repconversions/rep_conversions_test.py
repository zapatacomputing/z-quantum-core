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
"""Tests for conversions.py"""
import unittest

import numpy
from zquantum.core.openfermion.chem import MolecularData
from zquantum.core.openfermion.config import EQ_TOLERANCE
from zquantum.core.openfermion.ops.operators import FermionOperator
from zquantum.core.openfermion.ops.representations import InteractionOperatorError
from zquantum.core.openfermion.transforms.opconversions import get_fermion_operator
from zquantum.core.openfermion.transforms.opconversions.term_reordering import (
    normal_ordered,
)
from zquantum.core.openfermion.transforms.repconversions.conversions import (
    get_interaction_operator,
    get_molecular_data,
)


class GetInteractionOperatorTest(unittest.TestCase):
    def test_get_molecular_operator(self):
        coefficient = 3.0
        operators = ((2, 1), (3, 0), (0, 0), (3, 1))
        op = FermionOperator(operators, coefficient)

        molecular_operator = get_interaction_operator(op)
        fermion_operator = get_fermion_operator(molecular_operator)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(normal_ordered(op) == fermion_operator)

        op = FermionOperator("1^ 1")
        op *= 0.5 * EQ_TOLERANCE
        molecular_operator = get_interaction_operator(op)
        self.assertEqual(molecular_operator.constant, 0)
        self.assertTrue(
            numpy.allclose(molecular_operator.one_body_tensor, numpy.zeros((2, 2)))
        )

    def test_get_interaction_operator_bad_input(self):
        with self.assertRaises(TypeError):
            get_interaction_operator("3")

    def test_get_interaction_operator_below_threshold(self):
        op = get_interaction_operator(FermionOperator("1^ 1", 0.0))
        self.assertEqual(op.constant, 0)
        self.assertTrue(numpy.allclose(op.one_body_tensor, numpy.zeros((1, 1))))
        self.assertTrue(numpy.allclose(op.two_body_tensor, numpy.zeros((1, 1, 1, 1))))

    def test_get_interaction_operator_too_few_qubits(self):
        with self.assertRaises(ValueError):
            get_interaction_operator(FermionOperator("3^ 2^ 1 0"), 3)

    def test_get_interaction_operator_bad_1body_term(self):
        with self.assertRaises(InteractionOperatorError):
            get_interaction_operator(FermionOperator("1^ 0^"))

    def test_get_interaction_operator_bad_2body_term(self):
        with self.assertRaises(InteractionOperatorError):
            get_interaction_operator(FermionOperator("3^ 2 1 0"))

    def test_get_interaction_operator_nonmolecular_term(self):
        with self.assertRaises(InteractionOperatorError):
            get_interaction_operator(FermionOperator("3^ 2 1"))

    def test_get_molecular_data(self):
        """Test conversion to MolecularData from InteractionOperator"""
        coefficient = 3.0
        operators = ((2, 1), (3, 0), (0, 0), (3, 1))
        op = FermionOperator(operators, coefficient)

        molecular_operator = get_interaction_operator(op)
        molecule = get_molecular_data(
            molecular_operator,
            geometry=[["H", [0, 0, 0]]],
            basis="aug-cc-pvtz",
            multiplicity=2,
            n_electrons=1,
        )
        self.assertTrue(isinstance(molecule, MolecularData))

        molecule = get_molecular_data(
            molecular_operator,
            geometry=[["H", [0, 0, 0]]],
            basis="aug-cc-pvtz",
            multiplicity=2,
            n_electrons=1,
            reduce_spin=False,
        )
        self.assertTrue(isinstance(molecule, MolecularData))
