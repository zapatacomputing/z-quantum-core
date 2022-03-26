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

import pytest
import sympy
from zquantum.core.openfermion.ops.operators import FermionOperator, QubitOperator
from zquantum.core.openfermion.transforms.opconversions.conversions import (
    check_no_sympy,
    get_fermion_operator,
)


def test_get_fermion_operator_wrong_type():
    with pytest.raises(TypeError):
        _ = get_fermion_operator(QubitOperator())


class RaisesSympyExceptionTest(unittest.TestCase):
    def test_raises_sympy_expression(self):
        operator = FermionOperator("0^", sympy.Symbol("x"))
        with self.assertRaises(TypeError):
            check_no_sympy(operator)
