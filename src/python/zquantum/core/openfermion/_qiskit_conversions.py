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

"""
Translates OpenFermion Objects to qiskit SummedOp objects
"""
from openfermion import QubitOperator, count_qubits
from qiskit.opflow import PauliOp, SummedOp
from qiskit.quantum_info import Pauli


def qubitop_to_qiskitpauli(qubit_operator: QubitOperator) -> SummedOp:
    """Convert a OpenFermion QubitOperator to a SummedOp.

    Args:
        qubit_operator: OpenFermion QubitOperator to convert

    Returns:
        SummedOp representing the qubit operator
    """
    if not isinstance(qubit_operator, QubitOperator):
        raise TypeError("qubit_operator must be an OpenFermion QubitOperator object")

    terms = []
    for qubit_terms, coefficient in qubit_operator.terms.items():
        string_term = "I" * count_qubits(qubit_operator)
        for i, (term_qubit, term_pauli) in enumerate(qubit_terms):
            string_term = (
                string_term[:term_qubit] + term_pauli + string_term[term_qubit + 1 :]
            )
        terms.append(PauliOp(Pauli.from_label(string_term), coeff=coefficient))

    return SummedOp(terms)


def qiskitpauli_to_qubitop(qiskit_pauli: SummedOp) -> QubitOperator:
    """Convert a qiskit's SummedOp to an OpenFermion QubitOperator.

    Args:
        qiskit_pauli: operator to convert

    Returns:
        QubitOperator representing the SummedOp
    """

    if not isinstance(qiskit_pauli, SummedOp):
        raise TypeError("qiskit_pauli must be a qiskit SummedOp")

    transformed_term = QubitOperator()

    for pauli_op in qiskit_pauli._oplist:
        qiskit_term, weight = pauli_op.primitive, pauli_op.coeff

        openfermion_term = QubitOperator()
        for (term_qubit, term_pauli) in enumerate(str(qiskit_term)):
            if term_pauli != "I":
                if openfermion_term == QubitOperator():
                    openfermion_term = QubitOperator(f"[{term_pauli}{term_qubit}]")
                else:
                    openfermion_term *= QubitOperator(f"[{term_pauli}{term_qubit}]")

        transformed_term += openfermion_term * weight

    return transformed_term
