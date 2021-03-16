############################################################################
#   Copyright 2017 Rigetti Computing, Inc.
#   Modified by Zapata Computing 2020 to work for qiskit's WeightedPauliOperator.
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
############################################################################

import pytest
import numpy as np
from openfermion.ops import (
    FermionOperator,
    QubitOperator,
    InteractionOperator,
    InteractionRDM,
)
from openfermion.utils import hermitian_conjugated
from openfermion.transforms import jordan_wigner
from ._qiskit_conversions import (
    qiskitpauli_to_qubitop,
    qubitop_to_qiskitpauli,
)
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli


def test_translation_type_enforcement():
    """
    Make sure type check works
    """
    create_one = FermionOperator("1^")
    empty_one_body = np.zeros((2, 2))
    empty_two_body = np.zeros((2, 2, 2, 2))
    interact_one = InteractionOperator(1, empty_one_body, empty_two_body)
    interact_rdm = InteractionRDM(empty_one_body, empty_two_body)

    with pytest.raises(TypeError):
        qubitop_to_qiskitpauli(create_one)
    with pytest.raises(TypeError):
        qubitop_to_qiskitpauli(interact_one)
    with pytest.raises(TypeError):
        qubitop_to_qiskitpauli(interact_rdm)


def test_qubitop_to_qiskitpauli():
    """
    Conversion of QubitOperator; accuracy test
    """
    hop_term = FermionOperator(((2, 1), (0, 0)))
    term = hop_term + hermitian_conjugated(hop_term)

    pauli_term = jordan_wigner(term)

    qiskit_op = qubitop_to_qiskitpauli(pauli_term)

    ground_truth = WeightedPauliOperator(
        [[0.5, Pauli.from_label("XZX")], [0.5, Pauli.from_label("YZY")]]
    )

    assert ground_truth == qiskit_op


def test_qubitop_to_qiskitpauli_zero():
    zero_term = QubitOperator()
    qiskit_term = qubitop_to_qiskitpauli(zero_term)
    ground_truth = WeightedPauliOperator([])

    assert ground_truth == qiskit_term


def test_qiskitpauli_to_qubitop():
    qiskit_term = WeightedPauliOperator(
        [
            [1, Pauli.from_label("XIIIIY")],
        ]
    )

    op_fermion_term = QubitOperator(((0, "X"), (5, "Y")))
    test_op_fermion_term = qiskitpauli_to_qubitop(qiskit_term)

    assert test_op_fermion_term.isclose(op_fermion_term)


def test_qiskitpauli_to_qubitop_type_enforced():
    """Enforce the appropriate type"""
    create_one = FermionOperator("1^")
    empty_one_body = np.zeros((2, 2))
    empty_two_body = np.zeros((2, 2, 2, 2))
    interact_one = InteractionOperator(1, empty_one_body, empty_two_body)
    interact_rdm = InteractionRDM(empty_one_body, empty_two_body)

    with pytest.raises(TypeError):
        qiskitpauli_to_qubitop(create_one)
    with pytest.raises(TypeError):
        qiskitpauli_to_qubitop(interact_one)
    with pytest.raises(TypeError):
        qiskitpauli_to_qubitop(interact_rdm)
