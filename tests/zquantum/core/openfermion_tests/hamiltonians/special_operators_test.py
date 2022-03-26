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
"""testing angular momentum generators. _fermion_spin_operators.py"""
import unittest

from zquantum.core.openfermion.hamiltonians.special_operators import number_operator
from zquantum.core.openfermion.ops.operators import FermionOperator
from zquantum.core.openfermion.transforms.opconversions import normal_ordered
from zquantum.core.openfermion.utils import commutator


class NumberOperatorTest(unittest.TestCase):
    def test_fermion_number_operator_site(self):
        op = number_operator(3, 2, 1j, -1)
        self.assertEqual(op, FermionOperator(((2, 1), (2, 0))) * 1j)

    def test_number_operator_nosite(self):
        op = number_operator(4, parity=-1)
        expected = (
            FermionOperator(((0, 1), (0, 0)))
            + FermionOperator(((1, 1), (1, 0)))
            + FermionOperator(((2, 1), (2, 0)))
            + FermionOperator(((3, 1), (3, 0)))
        )
        self.assertEqual(op, expected)

    def test_bad_parity(self):
        with self.assertRaises(ValueError):
            number_operator(4, parity=2)
