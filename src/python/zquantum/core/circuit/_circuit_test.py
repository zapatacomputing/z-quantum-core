import unittest
import numpy as np
import random
from textwrap import dedent
import pyquil
import cirq
import qiskit
import os

from pyquil import Program
from pyquil.gates import *
from pyquil.gates import QUANTUM_GATES
from math import pi, sin, cos

from . import load_circuit, save_circuit, Circuit, Gate, Qubit, pyquil2cirq, cirq2pyquil


from ..utils import compare_unitary, is_identity, is_unitary, RNDSEED
from ..testing import create_random_circuit


class TestCircuit(unittest.TestCase):
    def test_circuit_eq(self):
        """Test equality operation between Circuit objects.
        """

        qubits = [Qubit(i) for i in range(0, 3)]
        gate_H0 = Gate("H", [qubits[0]])
        gate_CNOT01 = Gate("CNOT", [qubits[0], qubits[1]])
        gate_T2 = Gate("T", [qubits[2]])
        gate_CZ12 = Gate("CZ", [qubits[1], qubits[2]])

        circ1 = Circuit("circ1")
        circ1.qubits = qubits
        circ1.gates = [gate_H0, gate_CNOT01, gate_T2]

        circ2 = Circuit("circ2")
        circ2.qubits = qubits
        circ2.gates = [gate_H0, gate_CNOT01, gate_CZ12]

        circ3 = Circuit("circ3")
        circ3.qubits = qubits
        circ3.gates = [gate_H0, gate_CNOT01, gate_T2]

        self.assertEqual(circ1 == circ2, False)
        self.assertEqual(circ1 == circ3, True)

    def test_circuit_add(self):
        """Test addition operation between Circuit objects.
        """

        qubits = [Qubit(i) for i in range(0, 3)]
        gate_H0 = Gate("H", [qubits[0]])
        gate_CNOT01 = Gate("CNOT", [qubits[0], qubits[1]])
        gate_T2 = Gate("T", [qubits[2]])
        gate_CZ12 = Gate("CZ", [qubits[1], qubits[2]])

        circ1 = Circuit("circ1")
        circ1.qubits = qubits
        circ1.gates = [gate_H0, gate_CNOT01]

        circ2 = Circuit("circ2")
        circ2.qubits = qubits
        circ2.gates = [gate_T2, gate_CZ12]

        circ3 = Circuit("circ3")
        circ3.qubits = qubits
        circ3.gates = [gate_H0, gate_CNOT01, gate_T2, gate_CZ12]

        self.assertEqual(circ1 + circ2, circ3)

    def test_circuit_init(self):
        pyquil_prog = pyquil.quil.Program().inst(
            pyquil.gates.X(0),
            pyquil.gates.T(1),
            pyquil.gates.CNOT(0, 1),
            pyquil.gates.SWAP(0, 1),
            pyquil.gates.CZ(1, 0),
        )

        qubits = [cirq.LineQubit(i) for i in pyquil_prog.get_qubits()]

        gates = [
            cirq.X(qubits[0]),
            cirq.T(qubits[1]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.SWAP(qubits[0], qubits[1]),
            cirq.CZ(qubits[1], qubits[0]),
        ]

        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(gates, strategy=cirq.circuits.InsertStrategy.EARLIEST)

        circuit_from_cirq = Circuit(cirq_circuit)
        circuit_from_pyquil = Circuit(pyquil_prog)

        self.assertEqual(circuit_from_cirq, circuit_from_pyquil)

    def test_circuit_io(self):
        circuit = Circuit(Program().inst(X(0), Y(1), Z(0)))
        save_circuit(circuit, "circuit.json")
        loaded_circuit = load_circuit("circuit.json")
        self.assertTrue(circuit == loaded_circuit)
        os.remove("circuit.json")

    def test_zxz_cirq(self):
        """Test the special gate ZXZ (from cirq PhasedXPowGate)
        """

        random.seed(RNDSEED)
        beta = random.uniform(-1, 1)
        gamma = random.uniform(-1, 1)
        cirq_gate = cirq.PhasedXPowGate(phase_exponent=beta, exponent=gamma)
        cirq_circuit = cirq.Circuit()
        q = cirq.LineQubit(0)
        cirq_circuit.append(
            [cirq_gate(q)], strategy=cirq.circuits.InsertStrategy.EARLIEST
        )
        circuit2 = Circuit(cirq_circuit)
        circuit3 = circuit2.to_cirq()

        U1 = Circuit(cirq2pyquil(cirq_circuit)).to_unitary()
        U2 = Circuit(cirq2pyquil(circuit3)).to_unitary()
        u = np.array(
            [
                [
                    cos(gamma * pi / 2),
                    -sin(beta * pi) * sin(gamma * pi / 2)
                    - 1j * cos(beta * pi) * sin(gamma * pi / 2),
                ],
                [
                    sin(beta * pi) * sin(gamma * pi / 2)
                    - 1j * cos(beta * pi) * sin(gamma * pi / 2),
                    cos(gamma * pi / 2),
                ],
            ]
        )

        if compare_unitary(U1, U2) == False:
            print(cirq2pyquil(cirq_circuit))
            print(cirq2pyquil(circuit3))
        if compare_unitary(u, U1) == False:
            print(u)
            print(U1)
        self.assertTrue(compare_unitary(U1, U2, tol=1e-10))
        self.assertTrue(compare_unitary(u, U1, tol=1e-10))

    def test_zxz_pyquil(self):
        """Test the special gate ZXZ (from cirq PhasedXPowGate) for pyquil
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # rotation angles (in multiples of pi)
        gamma = random.uniform(-1, 1)  # rotation angles (in multiples of pi)
        circ1 = pyquil.quil.Program(
            RZ(-beta * pi, 0), RX(gamma * pi, 0), RZ(beta * pi, 0)
        )

        cirq_gate = cirq.PhasedXPowGate(phase_exponent=beta, exponent=gamma)
        cirq_circuit = cirq.Circuit()
        q = cirq.LineQubit(0)
        cirq_circuit.append(
            [cirq_gate(q)], strategy=cirq.circuits.InsertStrategy.EARLIEST
        )
        z_circuit = Circuit(cirq_circuit)
        circ2 = z_circuit.to_pyquil()

        # desired unitary
        u = np.array(
            [
                [
                    cos(gamma * pi / 2),
                    -sin(beta * pi) * sin(gamma * pi / 2)
                    - 1j * cos(beta * pi) * sin(gamma * pi / 2),
                ],
                [
                    sin(beta * pi) * sin(gamma * pi / 2)
                    - 1j * cos(beta * pi) * sin(gamma * pi / 2),
                    cos(gamma * pi / 2),
                ],
            ]
        )

        u1 = Circuit(circ1).to_unitary()
        u2 = Circuit(circ2).to_unitary()
        u3 = cirq_gate._unitary_()
        if compare_unitary(u1, u2, tol=1e-10) == False:
            print("u1={}".format(u1))
            print("u2={}".format(u2))
        if compare_unitary(u2, u3, tol=1e-10) == False:
            print("u2={}".format(u2))
            print("u3={}".format(u3))
        if compare_unitary(u3, u, tol=1e-10) == False:
            print("u2={}".format(u2))
            print("u3={}".format(u3))
        self.assertTrue(compare_unitary(u1, u2, tol=1e-10))
        self.assertTrue(compare_unitary(u2, u3, tol=1e-10))
        self.assertTrue(compare_unitary(u3, u, tol=1e-10))

    def test_zxz_qiskit(self):
        """Test the special gate ZXZ (from cirq PhasedXPowGate) for qiskit
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # rotation angles (in multiples of pi)
        gamma = random.uniform(-1, 1)  # rotation angles (in multiples of pi)

        qubits = qiskit.QuantumRegister(1)
        circ = qiskit.QuantumCircuit(qubits)
        circ.rz(-beta * pi, qubits[0])
        circ.rx(gamma * pi, qubits[0])
        circ.rz(beta * pi, qubits[0])
        z_circuit = Circuit(circ)
        u1 = z_circuit.to_unitary()

        circ2 = cirq.PhasedXPowGate(phase_exponent=beta, exponent=gamma)
        u2 = circ2._unitary_()
        self.assertTrue(compare_unitary(u1, u2, tol=1e-10))

    def test_rh_cirq(self):
        """Test the special RH (from cirq HPowGate) for cirq
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)
        cirq_gate = cirq.HPowGate(exponent=beta)
        cirq_circuit = cirq.Circuit()
        q = cirq.LineQubit(0)
        cirq_circuit.append(
            [cirq_gate(q)], strategy=cirq.circuits.InsertStrategy.EARLIEST
        )
        circuit2 = Circuit(cirq_circuit)
        circuit3 = circuit2.to_cirq()

        # desired operator
        elem00 = cos(beta * pi / 2) - 1j * 1 / np.sqrt(2) * sin(beta * pi / 2)
        elem01 = -1j * 1 / np.sqrt(2) * sin(beta * pi / 2)
        elem10 = -1j * 1 / np.sqrt(2) * sin(beta * pi / 2)
        elem11 = cos(beta * pi / 2) + 1j * 1 / np.sqrt(2) * sin(beta * pi / 2)
        u = np.array([[elem00, elem01], [elem10, elem11]])

        U1 = Circuit(cirq2pyquil(cirq_circuit)).to_unitary()
        U2 = Circuit(cirq2pyquil(circuit3)).to_unitary()

        if compare_unitary(U1, U2, tol=1e-10) == False:
            print(cirq2pyquil(cirq_circuit))
            print(cirq2pyquil(circuit3))
        self.assertTrue(compare_unitary(U1, U2, tol=1e-10), True)
        self.assertTrue(compare_unitary(u, U1, tol=1e-10), True)

    def test_rh_pyquil(self):
        """Test the special RH (from cirq HPowGate) for pyquil
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # rotation angles (in multiples of pi)
        circ1 = pyquil.quil.Program(RY(-pi / 4, 0), RZ(beta * pi, 0), RY(pi / 4, 0))

        cirq_gate = cirq.HPowGate(exponent=beta)
        cirq_circuit = cirq.Circuit()
        q = cirq.LineQubit(0)
        cirq_circuit.append(
            [cirq_gate(q)], strategy=cirq.circuits.InsertStrategy.EARLIEST
        )
        z_circuit = Circuit(cirq_circuit)
        circ2 = z_circuit.to_pyquil()

        u1 = Circuit(circ1).to_unitary()
        u2 = Circuit(circ2).to_unitary()
        u3 = cirq_gate._unitary_()

        # desired operator
        elem00 = cos(beta * pi / 2) - 1j * 1 / np.sqrt(2) * sin(beta * pi / 2)
        elem01 = -1j * 1 / np.sqrt(2) * sin(beta * pi / 2)
        elem10 = -1j * 1 / np.sqrt(2) * sin(beta * pi / 2)
        elem11 = cos(beta * pi / 2) + 1j * 1 / np.sqrt(2) * sin(beta * pi / 2)
        u = np.array([[elem00, elem01], [elem10, elem11]])

        self.assertTrue(compare_unitary(u1, u2, tol=1e-10))
        self.assertTrue(compare_unitary(u2, u3, tol=1e-10))
        self.assertTrue(compare_unitary(u3, u, tol=1e-10))

    def test_rh_qiskit(self):
        """Test the special RH (from cirq HPowGate) for qiskit
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # rotation angles (in multiples of pi)

        qubits = qiskit.QuantumRegister(1)
        circ = qiskit.QuantumCircuit(qubits)
        circ.ry(-pi / 4, qubits[0])
        circ.rz(beta * pi, qubits[0])
        circ.ry(pi / 4, qubits[0])
        z_circuit = Circuit(circ)
        u1 = z_circuit.to_unitary()

        circ2 = cirq.HPowGate(exponent=beta)
        u2 = circ2._unitary_()
        self.assertTrue(compare_unitary(u1, u2, tol=1e-10))

    def test_xx_cirq(self):
        """Test the special XX (modified from cirq XXPowGate) for cirq
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # we want to evolve under XX for time beta*pi
        cirq_gate = cirq.XXPowGate(exponent=beta * 2)
        cirq_circuit = cirq.Circuit()
        q = cirq.LineQubit(0)
        q2 = cirq.LineQubit(1)
        cirq_circuit.append(cirq_gate(q, q2))
        circuit2 = Circuit(cirq_circuit)
        circuit3 = circuit2.to_cirq()

        U1 = Circuit(cirq2pyquil(cirq_circuit)).to_unitary()
        U2 = Circuit(cirq2pyquil(circuit3)).to_unitary()
        U3 = [
            [cos(beta * pi), 0, 0, -1j * sin(beta * pi)],
            [0, cos(beta * pi), -1j * sin(beta * pi), 0],
            [0, -1j * sin(beta * pi), cos(beta * pi), 0],
            [-1j * sin(beta * pi), 0, 0, cos(beta * pi)],
        ]

        if compare_unitary(U1, U2, tol=1e-10) == False:
            print(U1)
            print(U2)
        if compare_unitary(U2, U3, tol=1e-10) == False:
            print(U2)
            print(U3)
        self.assertTrue(compare_unitary(U1, U2, tol=1e-10), True)
        self.assertTrue(compare_unitary(U2, U3, tol=1e-10), True)

    def test_xx_pyquil(self):
        """Test the special XX (modified from cirq XXPowGate) for pyquil
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # rotation angles (in multiples of pi)
        circ1 = pyquil.quil.Program(
            H(0), H(1), CNOT(0, 1), RZ(beta * 2 * pi, 1), CNOT(0, 1), H(0), H(1)
        )

        cirq_gate = cirq.XXPowGate(exponent=beta * 2)
        cirq_circuit = cirq.Circuit()
        q = cirq.LineQubit(0)
        q2 = cirq.LineQubit(1)
        cirq_circuit.append(cirq_gate(q, q2))
        z_circuit = Circuit(cirq_circuit)
        circ2 = z_circuit.to_pyquil()

        u1 = Circuit(circ1).to_unitary()
        u2 = Circuit(circ2).to_unitary()
        u3 = cirq_gate._unitary_()
        u4 = [
            [cos(beta * pi), 0, 0, -1j * sin(beta * pi)],
            [0, cos(beta * pi), -1j * sin(beta * pi), 0],
            [0, -1j * sin(beta * pi), cos(beta * pi), 0],
            [-1j * sin(beta * pi), 0, 0, cos(beta * pi)],
        ]

        self.assertTrue(compare_unitary(u1, u2, tol=1e-10))
        self.assertTrue(compare_unitary(u2, u3, tol=1e-10))
        self.assertTrue(compare_unitary(u3, u4, tol=1e-10))

    def test_xx_qiskit(self):
        """Test the special XX (modified from cirq XXPowGate) for qiskit
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # rotation angles (in multiples of pi)

        qubits = qiskit.QuantumRegister(2)
        circ = qiskit.QuantumCircuit(qubits)
        circ.h(qubits[0])
        circ.h(qubits[1])
        circ.cx(qubits[0], qubits[1])
        circ.rz(beta * 2 * pi, qubits[1])
        circ.cx(qubits[0], qubits[1])
        circ.h(qubits[0])
        circ.h(qubits[1])
        z_circuit = Circuit(circ)
        u1 = z_circuit.to_unitary()

        circ2 = cirq.XXPowGate(exponent=beta * 2)
        u2 = circ2._unitary_()
        u3 = [
            [cos(beta * pi), 0, 0, -1j * sin(beta * pi)],
            [0, cos(beta * pi), -1j * sin(beta * pi), 0],
            [0, -1j * sin(beta * pi), cos(beta * pi), 0],
            [-1j * sin(beta * pi), 0, 0, cos(beta * pi)],
        ]
        self.assertTrue(compare_unitary(u1, u2, tol=1e-10))
        self.assertTrue(compare_unitary(u2, u3, tol=1e-10))

    def test_yy_cirq(self):
        """Test the special YY (modified from cirq YYPowGate) for cirq
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # we want to evolve under XX for time beta*pi
        cirq_gate = cirq.YYPowGate(exponent=beta * 2)
        cirq_circuit = cirq.Circuit()
        q = cirq.LineQubit(0)
        q2 = cirq.LineQubit(1)
        cirq_circuit.append(cirq_gate(q, q2))
        circuit2 = Circuit(cirq_circuit)
        circuit3 = circuit2.to_cirq()

        U1 = Circuit(cirq2pyquil(cirq_circuit)).to_unitary()
        U2 = Circuit(cirq2pyquil(circuit3)).to_unitary()
        U3 = [
            [cos(beta * pi), 0, 0, 1j * sin(beta * pi)],
            [0, cos(beta * pi), -1j * sin(beta * pi), 0],
            [0, -1j * sin(beta * pi), cos(beta * pi), 0],
            [1j * sin(beta * pi), 0, 0, cos(beta * pi)],
        ]

        if compare_unitary(U1, U2, tol=1e-10) == False:
            print(U1)
            print(U2)
        if compare_unitary(U2, U3, tol=1e-10) == False:
            print(U2)
            print(U3)
        self.assertTrue(compare_unitary(U1, U2, tol=1e-10), True)
        self.assertTrue(compare_unitary(U2, U3, tol=1e-10), True)

    def test_yy_pyquil(self):
        """Test the special YY (modified from cirq YYPowGate) for pyquil
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # rotation angles (in multiples of pi)
        circ1 = pyquil.quil.Program(
            S(0),
            S(1),
            H(0),
            H(1),
            CNOT(0, 1),
            RZ(beta * 2 * pi, 1),
            CNOT(0, 1),
            H(0),
            H(1),
            RZ(-pi / 2, 0),
            RZ(-pi / 2, 1),
        )

        cirq_gate = cirq.YYPowGate(exponent=beta * 2)
        cirq_circuit = cirq.Circuit()
        q = cirq.LineQubit(0)
        q2 = cirq.LineQubit(1)
        cirq_circuit.append(cirq_gate(q, q2))
        z_circuit = Circuit(cirq_circuit)
        circ2 = z_circuit.to_pyquil()

        u1 = Circuit(circ1).to_unitary()
        u2 = Circuit(circ2).to_unitary()
        u3 = cirq_gate._unitary_()
        u4 = [
            [cos(beta * pi), 0, 0, 1j * sin(beta * pi)],
            [0, cos(beta * pi), -1j * sin(beta * pi), 0],
            [0, -1j * sin(beta * pi), cos(beta * pi), 0],
            [1j * sin(beta * pi), 0, 0, cos(beta * pi)],
        ]

        self.assertTrue(compare_unitary(u1, u2, tol=1e-10))
        self.assertTrue(compare_unitary(u2, u3, tol=1e-10))
        self.assertTrue(compare_unitary(u3, u4, tol=1e-10))

    def test_yy_qiskit(self):
        """Test the special YY (modified from cirq YYPowGate) for qiskit
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # rotation angles (in multiples of pi)

        qubits = qiskit.QuantumRegister(2)
        circ = qiskit.QuantumCircuit(qubits)
        circ.s(qubits[0])
        circ.s(qubits[1])
        circ.h(qubits[0])
        circ.h(qubits[1])
        circ.cx(qubits[0], qubits[1])
        circ.rz(beta * 2 * pi, qubits[1])
        circ.cx(qubits[0], qubits[1])
        circ.h(qubits[0])
        circ.h(qubits[1])
        circ.rz(-pi / 2, qubits[0])
        circ.rz(-pi / 2, qubits[1])
        z_circuit = Circuit(circ)
        u1 = z_circuit.to_unitary()

        circ2 = cirq.YYPowGate(exponent=beta * 2)
        u2 = circ2._unitary_()
        u3 = [
            [cos(beta * pi), 0, 0, 1j * sin(beta * pi)],
            [0, cos(beta * pi), -1j * sin(beta * pi), 0],
            [0, -1j * sin(beta * pi), cos(beta * pi), 0],
            [1j * sin(beta * pi), 0, 0, cos(beta * pi)],
        ]
        self.assertTrue(compare_unitary(u1, u2, tol=1e-10))
        self.assertTrue(compare_unitary(u2, u3, tol=1e-10))

    def test_zz_cirq(self):
        """Test the special ZZ (modified from cirq ZZPowGate) for cirq
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # we want to evolve under ZZ for time beta*pi
        cirq_gate = cirq.ZZPowGate(exponent=beta * 2)
        cirq_circuit = cirq.Circuit()
        q = cirq.LineQubit(0)
        q2 = cirq.LineQubit(1)
        cirq_circuit.append(cirq_gate(q, q2))
        circuit2 = Circuit(cirq_circuit)
        circuit3 = circuit2.to_cirq()

        U1 = Circuit(cirq2pyquil(cirq_circuit)).to_unitary()
        U2 = Circuit(cirq2pyquil(circuit3)).to_unitary()
        U3 = [
            [cos(beta * pi) - 1j * sin(beta * pi), 0, 0, 0],
            [0, cos(beta * pi) + 1j * sin(beta * pi), 0, 0],
            [0, 0, cos(beta * pi) + 1j * sin(beta * pi), 0],
            [0, 0, 0, cos(beta * pi) - 1j * sin(beta * pi)],
        ]

        if compare_unitary(U1, U2, tol=1e-10) == False:
            print(U1)
            print(U2)
        if compare_unitary(U2, U3, tol=1e-10) == False:
            print(U2)
            print(U3)
        self.assertTrue(compare_unitary(U1, U2, tol=1e-10), True)
        self.assertTrue(compare_unitary(U2, U3, tol=1e-10), True)

    def test_zz_pyquil(self):
        """Test the special ZZ (modified from cirq ZZPowGate) for pyquil
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # rotation angles (in multiples of pi)
        circ1 = pyquil.quil.Program(CNOT(0, 1), RZ(beta * 2 * pi, 1), CNOT(0, 1))

        cirq_gate = cirq.ZZPowGate(exponent=beta * 2)
        cirq_circuit = cirq.Circuit()
        q = cirq.LineQubit(0)
        q2 = cirq.LineQubit(1)
        cirq_circuit.append(cirq_gate(q, q2))
        z_circuit = Circuit(cirq_circuit)
        circ2 = z_circuit.to_pyquil()

        u1 = Circuit(circ1).to_unitary()
        u2 = Circuit(circ2).to_unitary()
        u3 = cirq_gate._unitary_()
        u4 = [
            [cos(beta * pi) - 1j * sin(beta * pi), 0, 0, 0],
            [0, cos(beta * pi) + 1j * sin(beta * pi), 0, 0],
            [0, 0, cos(beta * pi) + 1j * sin(beta * pi), 0],
            [0, 0, 0, cos(beta * pi) - 1j * sin(beta * pi)],
        ]

        self.assertTrue(compare_unitary(u1, u2, tol=1e-10))
        self.assertTrue(compare_unitary(u2, u3, tol=1e-10))
        self.assertTrue(compare_unitary(u3, u4, tol=1e-10))

    def test_zz_qiskit(self):
        """Test the special ZZ (modified from cirq ZZPowGate) for qiskit
        """

        # random.seed(RNDSEED)
        beta = random.uniform(-1, 1)  # rotation angles (in multiples of pi)

        qubits = qiskit.QuantumRegister(2)
        circ = qiskit.QuantumCircuit(qubits)

        circ.cx(qubits[0], qubits[1])
        circ.rz(beta * 2 * pi, qubits[1])
        circ.cx(qubits[0], qubits[1])

        z_circuit = Circuit(circ)
        u1 = z_circuit.to_unitary()

        circ2 = cirq.ZZPowGate(exponent=beta * 2)
        u2 = circ2._unitary_()
        u3 = [
            [cos(beta * pi) - 1j * sin(beta * pi), 0, 0, 0],
            [0, cos(beta * pi) + 1j * sin(beta * pi), 0, 0],
            [0, 0, cos(beta * pi) + 1j * sin(beta * pi), 0],
            [0, 0, 0, cos(beta * pi) - 1j * sin(beta * pi)],
        ]
        self.assertTrue(compare_unitary(u1, u2, tol=1e-10))
        self.assertTrue(compare_unitary(u2, u3, tol=1e-10))

    def test_pyquil2cirq(self):
        qprog = pyquil.quil.Program().inst(
            pyquil.gates.X(0),
            pyquil.gates.T(1),
            pyquil.gates.RZ(0.1, 3),
            pyquil.gates.CNOT(0, 1),
            pyquil.gates.SWAP(0, 1),
            pyquil.gates.CZ(1, 0),
        )

        circuit = pyquil2cirq(qprog)
        qubits = [cirq.GridQubit(i, 0) for i in qprog.get_qubits()]

        gates = [
            cirq.X(qubits[0]),
            cirq.T(qubits[1]),
            cirq.Rz(rads=0.1)(qubits[2]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.SWAP(qubits[0], qubits[1]),
            cirq.CZ(qubits[1], qubits[0]),
        ]

        ref_circuit = cirq.Circuit()
        ref_circuit.append(gates, strategy=cirq.circuits.InsertStrategy.EARLIEST)

        # Get the unitary matrix for the circuit
        U = circuit.unitary()

        # Get the unitary matrix for the reference circuit
        U_ref = ref_circuit.unitary()

        # Check that the matrices are identical to within a global phase. See J. Chem. Phys. 134, 144112 (2011).
        self.assertTrue(compare_unitary(U, U_ref))

    def test_pyquil_empty(self):
        """Convert empty pyquil Program object to Circuit object and back.
        """

        prog = pyquil.Program()
        p = Circuit(prog)
        prog2 = p.to_pyquil()
        self.assertEqual(prog, prog2)

    def test_pyquil_conversion_specific(self):
        """The goal of this test is to probe if the conversion between core.circuit.Circuit 
        and pyquil Program object is seamless, restricted to the gate set hard coded in pyquil. The
        test program will build a specific quantum circuit. At the end the assertion compares 
        the original pyquil Program with the pyquil Program produced by converting to QCircuit 
        and back to pyquil.
        """

        prog = pyquil.Program(
            H(0), CNOT(0, 1), RX(0.5, 2), CPHASE(0.6, 2, 3), SWAP(1, 2), RY(0.7, 1)
        )
        qc = Circuit(prog)
        prog2 = qc.to_pyquil()
        self.assertEqual(prog, prog2)

    def test_pyquil_conversion_general(self):
        """The goal of this test is to probe if the conversion between core.circuit.Circuit
        and pyquil Program object is seamless, restricted to the gate set hard coded in pyquil. The
        test program will randomly choose a number of qubits and a number of gates from 
        specified ranges, and proceed to generate a circuit where at each step a gate is
        uniformly randomly drawn from the set of all one-qubit, two-qubit and three-qubit
        gates specified in pyquil. At the end the assertion compares the original pyquil
        Program with the pyquil Program produced by converting to QCircuit and back to
        pyquil.
        """

        onequbit_gates = [
            #'I',
            "X",
            "Y",
            "Z",
            "H",
            "S",
            "T",
            "RX",
            "RY",
            "RZ",
            "PHASE",
        ]
        twoqubit_gates = [
            "CNOT",
            "CZ",
            # 'CPHASE00','CPHASE01','CPHASE10',
            "CPHASE",
            "SWAP",
            # 'ISWAP','PSWAP'
        ]
        threequbit_gates = [
            # 'CCNOT','CSWAP'
        ]

        all_gates = onequbit_gates + twoqubit_gates + threequbit_gates
        noparam_gates = [
            #'I',
            "X",
            "Y",
            "Z",
            "H",
            "S",
            "T",
            "CNOT",
            "CZ",
            "SWAP",
            #'ISWAP','CCNOT','CSWAP'
        ]
        oneparam_gates = [
            "RX",
            "RY",
            "RZ",
            "PHASE",
            #'CPHASE00','CPHASE01','CPHASE10',
            "CPHASE",
            #'PSWAP'
        ]

        nqubit_max = 10
        ngates_max = 40

        ntrials = 1
        # random.seed(RNDSEED)
        nqubits = random.randint(4, nqubit_max + 1)  # number of qubits
        ngates = random.randint(1, ngates_max + 1)  # number of gates

        prog = Program()

        for shot in range(0, ntrials):
            for i in range(0, ngates):
                gate = random.choice(all_gates)
                func = QUANTUM_GATES[gate]  # from gates.py in pyquil/

                if gate in onequbit_gates:
                    q = random.randint(0, nqubits - 1)
                    if gate in oneparam_gates:
                        theta = random.uniform(0, 2 * pi)
                        prog = prog + Program(func(theta, q))
                    if gate in noparam_gates:
                        prog = prog + Program(func(q))
                if gate in twoqubit_gates:
                    qindices = random.sample(set(range(0, nqubits)), 2)
                    q1 = qindices[0]
                    q2 = qindices[1]
                    if gate in oneparam_gates:
                        theta = random.uniform(0, 2 * pi)
                        prog = prog + Program(func(theta, q1, q2))
                    if gate in noparam_gates:
                        prog = prog + Program(func(q1, q2))
                if gate in threequbit_gates:
                    qindices = random.sample(set(range(0, nqubits)), 3)
                    q1 = qindices[0]
                    q2 = qindices[1]
                    q3 = qindices[2]
                    prog = prog + Program(func(q1, q2, q3))

            qc = Circuit(prog)
            prog2 = qc.to_pyquil()
            if prog != prog2:
                print(prog)
                print(prog2)
            self.assertEqual(prog, prog2)

    def test_cirq_empty(self):
        """Convert empty cirq Circuit object to core.circuit.Circuit object.
        """

        cirq_circuit = cirq.Circuit()
        p = Circuit(cirq_circuit)
        cirq_circuit2 = p.to_cirq()
        self.assertEqual(cirq_circuit, cirq_circuit2)

    def test_cirq_conversion_specific(self):
        """The goal of this test is to probe if the conversion between core.circuit.Circuit
        and cirq Circuit object is seamless, restricted to the gate set. 
        The test program will build a specific quantum circuit. At the end the assertion compares 
        the original cirq Circuit with the cirq Circuit produced by converting to Zap OS Circuit 
        and back to cirq.
        """

        qubits = [cirq.LineQubit(i) for i in range(0, 3)]

        gates = [
            cirq.X(qubits[0]),
            cirq.Z(qubits[1]) ** 0.7,
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.SWAP(qubits[1], qubits[2]),
            cirq.Y(qubits[2]),
            cirq.CZ(qubits[2], qubits[0]),
        ]

        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(gates, strategy=cirq.circuits.InsertStrategy.EARLIEST)

        qc = Circuit(cirq_circuit)
        cirq_circuit2 = qc.to_cirq()
        self.assertEqual(cirq_circuit, cirq_circuit2)

    def test_cirq_conversion_on_qubits(self):
        """For testing conversion to cirq circuits acting on a given set of cirq qubits.
        """

        qubits = [cirq.LineQubit(x) for x in range(0, 4)]
        gates = [cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[2]), cirq.H(qubits[2])]
        p1 = cirq.Circuit()
        p1.append(gates, strategy=cirq.circuits.InsertStrategy.EARLIEST)

        p2 = pyquil.Program(H(0), CNOT(1, 0), H(1))
        cirq_qubits = [qubits[2], qubits[0]]
        p2_cirq = Circuit(p2).to_cirq(cirq_qubits)

        test_circuit = p1 + p2_cirq
        self.assertEqual(
            is_identity(Circuit(test_circuit).to_unitary(), tol=1e-10), True
        )

    def test_cirq_conversion_general(self):
        """The goal of this test is to probe if the conversion between core.circuit.Circuit
        and cirq Circuit object is seamless, restricted to the gate set hard coded in cirq. The
        test program will randomly choose a number of qubits and a number of gates from 
        specified ranges, and proceed to generate a circuit where at each step a gate is
        uniformly randomly drawn from the set of all one-qubit, two-qubit and three-qubit
        gates specified in cirq. At the end the assertion compares the original cirq
        Circuit with the cirq Circuit produced by converting to Zap OS Circuit and back to
        cirq.
        """

        onequbit_gates = [
            "X",
            "Y",
            "Z",
            "H",
            "S",
            "T",
            "Rx",
            "Ry",
            "Rz",
            "PHASE",
            "ZXZ",
            "RH",
        ]
        twoqubit_gates = ["CNOT", "CZ", "CPHASE", "SWAP", "XX", "YY", "ZZ"]
        threequbit_gates = []

        all_gates = onequbit_gates + twoqubit_gates + threequbit_gates
        noparam_gates = ["X", "Y", "Z", "H", "S", "T", "CNOT", "CZ", "SWAP"]
        oneparam_gates = ["Rx", "Ry", "Rz", "PHASE", "CPHASE", "RH", "XX", "YY", "ZZ"]
        twoparam_gates = ["ZXZ"]

        nqubit_max = 6
        nqubit_min = 2
        ngates_min = 5
        ngates_max = 15

        ntrials = 10
        # random.seed(RNDSEED)
        nqubits = random.randint(nqubit_min, nqubit_max)  # number of qubits
        ngates = random.randint(ngates_min, ngates_max)  # number of gates

        op_map = {
            "X": cirq.X,
            "Y": cirq.Y,
            "Z": cirq.Z,
            "T": cirq.T,
            "H": cirq.H,
            "S": cirq.S,
            "Rx": cirq.Rx,
            "Ry": cirq.Ry,
            "Rz": cirq.Rz,
            "PHASE": cirq.Z,
            "ZXZ": cirq.PhasedXPowGate,
            "RH": cirq.H,
            "CNOT": cirq.CNOT,
            "SWAP": cirq.SWAP,
            "CZ": cirq.CZ,
            "CPHASE": cirq.ops.common_gates.CZPowGate,
            "XX": cirq.XX,
            "YY": cirq.YY,
            "ZZ": cirq.ZZ,
        }

        cirq_circuit = cirq.Circuit()

        _gates = []
        for shot in range(0, ntrials):
            for i in range(0, ngates):
                gate = random.choice(all_gates)
                func = op_map[gate]  # from gates.py in pyquil/

                if gate in onequbit_gates:
                    q = random.randint(0, nqubits - 1)
                    theta = random.uniform(-1, 1)
                    theta2 = random.uniform(-1, 1)
                    if gate in twoparam_gates:
                        if gate in {"ZXZ"}:
                            _gates.append(
                                func(phase_exponent=theta, exponent=theta2)(
                                    cirq.LineQubit(q)
                                )
                            )
                    if gate in oneparam_gates:
                        if gate in {"PHASE", "RH"}:
                            func = func ** theta
                            _gates.append(func(cirq.LineQubit(q)))
                        if gate in {"Rx", "Ry", "Rz"}:
                            _gates.append(func(theta)(cirq.LineQubit(q)))
                    if gate in noparam_gates:
                        _gates.append(func(cirq.LineQubit(q)))
                if gate in twoqubit_gates:
                    qindices = random.sample(set(range(0, nqubits)), 2)
                    q1 = qindices[0]
                    q2 = qindices[1]
                    # if gate in oneparam_gates:
                    #     theta = random.uniform(0,2*pi)
                    #     prog = prog + Program(func(theta,q1,q2))
                    if gate in noparam_gates:
                        _gates.append(func(cirq.LineQubit(q1), cirq.LineQubit(q2)))
                    if gate in oneparam_gates:
                        theta = random.uniform(-1, 1)
                        if gate in {"XX", "YY", "ZZ"}:
                            _gates.append(
                                func(cirq.LineQubit(q1), cirq.LineQubit(q2)) ** theta
                            )
                # if gate in threequbit_gates:
                #     qindices = random.sample(set(range(0, nqubits)),3)
                #     q1 = qindices[0]
                #     q2 = qindices[1]
                #     q3 = qindices[2]
                #     prog = prog + Program(func(q1,q2,q3))

            cirq_circuit = cirq.Circuit()
            cirq_circuit.append(_gates, strategy=cirq.circuits.InsertStrategy.EARLIEST)

            qc = Circuit(cirq_circuit)
            cirq_circuit2 = qc.to_cirq()

            U1 = Circuit(cirq2pyquil(cirq_circuit)).to_unitary()
            U2 = Circuit(cirq2pyquil(cirq_circuit2)).to_unitary()

            if compare_unitary(U1, U2, tol=1e-10) == False:
                print(cirq2pyquil(cirq_circuit))
                print(cirq2pyquil(cirq_circuit2))

            self.assertEqual(compare_unitary(U1, U2, tol=1e-10), True)

    def test_qiskit_empty(self):
        """Converting empty qiskit QuantumCircuit to and from core.circuit.Circuit objects.
        """
        qubits = qiskit.QuantumRegister(3)
        circ = qiskit.QuantumCircuit(qubits)

        # test if two qiskit QuantumCircuit objects are equal
        def test_equal_qiskit(circ, circ2):
            if len(circ.data) != len(circ2.data):
                return False
            for gate, gate2 in zip(circ.data, circ2.data):
                if gate != gate2:
                    return False
                else:
                    for qubit, qubit2 in zip(gate.qargs, gate2.qargs):
                        if qubit != qubit2:
                            return False
            return True

        p = Circuit(circ)
        circ2 = p.to_qiskit()
        self.assertEqual(test_equal_qiskit(circ, circ2), True)

    def test_qiskit_conversion_specific(self):
        """The goal of this test is to probe if the conversion between core.circuit.Circuit
        and qiskit QuantumCircuit object is seamless, restricted to the gate set. 
        The test program will build a specific quantum circuit. At the end the assertion compares 
        the original qiskit QuantumCircuit with the qiskit QuantumCircuit produced by converting to Zap OS Circuit 
        and back to qiskit.
        """

        qubits = qiskit.QuantumRegister(3)
        qubits2 = qiskit.QuantumRegister(2)

        circ = qiskit.QuantumCircuit(qubits, qubits2)
        circ.h(qubits[0])
        circ.cx(qubits[0], qubits2[1])
        circ.ry(0.6, qubits2[1])
        circ.cx(qubits2[1], qubits[2])
        circ.t(qubits[0])

        p = Circuit(circ)
        circ2 = p.to_qiskit()

        # test if two qiskit QuantumCircuit objects are equal
        def test_equal_qiskit(circ, circ2):
            if len(circ.data) != len(circ2.data):
                return False
            for gate, gate2 in zip(circ.data, circ2.data):
                if gate != gate2:
                    return False
                else:
                    for qubit, qubit2 in zip(gate[1], gate2[1]):
                        if qubit != qubit2:
                            return False
            return True

        self.assertEqual(test_equal_qiskit(circ, circ2), True)

    def test_qiskit_conversion_general(self):
        """The goal of this test is to probe if the conversion between core.circuit.Circuit
        and qiskit QuantumCircuit object is seamless, restricted to the gate set hard coded in qiskit. The
        test program will randomly choose a number of qubits and a number of gates from 
        specified ranges, and proceed to generate a circuit where at each step a gate is
        uniformly randomly drawn from the set of all one-qubit, two-qubit and three-qubit
        gates specified in qiskit. At the end the assertion compares the original qiskit
        QuantumCircuit with the qiskit Circuit produced by converting to Zap OS Circuit and back to
        qiskit.
        """

        onequbit_gates = ["X", "Y", "Z", "H", "S", "T", "Rx", "Ry", "Rz", "PHASE"]
        twoqubit_gates = [
            "CNOT",
            "CZ",
            #'CPHASE', leaving out CPHASE because this needs custom compiling
            "SWAP",
        ]
        threequbit_gates = []

        all_gates = onequbit_gates + twoqubit_gates + threequbit_gates
        noparam_gates = ["X", "Y", "Z", "H", "S", "T", "CNOT", "CZ", "SWAP"]
        oneparam_gates = [
            "Rx",
            "Ry",
            "Rz",
            "PHASE",
            # 'CPHASE'
        ]

        nqubit_max = 10
        ngates_max = 15

        ntrials = 10
        # random.seed(RNDSEED)
        nqubits = random.randint(4, nqubit_max + 1)  # number of qubits
        ngates = random.randint(1, ngates_max + 1)  # number of gates

        op_map = {
            "X": qiskit.extensions.standard.XGate,
            "Y": qiskit.extensions.standard.YGate,
            "Z": qiskit.extensions.standard.ZGate,
            "T": qiskit.extensions.standard.TGate,
            "H": qiskit.extensions.standard.HGate,
            "S": qiskit.extensions.standard.SGate,
            "Rx": qiskit.extensions.standard.RXGate,
            "Ry": qiskit.extensions.standard.RYGate,
            "Rz": qiskit.extensions.standard.RZGate,
            "PHASE": qiskit.extensions.standard.RZGate,
            "CNOT": qiskit.extensions.standard.CnotGate,
            "SWAP": qiskit.extensions.standard.SwapGate,
            "CZ": qiskit.extensions.standard.CzGate,
            #'CPHASE' : cirq.ops.common_gates.CZPowGate
        }

        qubits = qiskit.QuantumRegister(nqubits)
        circ = qiskit.QuantumCircuit(qubits)

        for _ in range(0, ntrials):
            for _ in range(0, ngates):
                gate = random.choice(all_gates)
                func = op_map[gate]  # from gates.py in pyquil/

                if gate in onequbit_gates:
                    q = random.randint(0, nqubits - 1)
                    if gate in oneparam_gates:
                        theta = random.uniform(0, 2 * pi)
                        circ.append(func(theta), qargs=[list(qubits)[q]])
                    if gate in noparam_gates:
                        circ.append(func(), qargs=[list(qubits)[q]])
                if gate in twoqubit_gates:
                    qindices = random.sample(set(range(0, nqubits)), 2)
                    q1 = qindices[0]
                    q2 = qindices[1]
                    if gate in oneparam_gates:
                        theta = random.uniform(0, 2 * pi)
                        circ.append(
                            func(theta), qargs=[list(qubits)[q1], list(qubits)[q2]]
                        )
                    if gate in noparam_gates:
                        circ.append(func(), qargs=[list(qubits)[q1], list(qubits)[q2]])
                # if gate in threequbit_gates:
                #     qindices = random.sample(set(range(0, nqubits)),3)
                #     q1 = qindices[0]
                #     q2 = qindices[1]
                #     q3 = qindices[2]
                #     prog = prog + Program(func(q1,q2,q3)

            p = Circuit(circ)
            circ2 = p.to_qiskit()

            # test if two qiskit QuantumCircuit objects are equal
            def test_equal_qiskit(circ, circ2):
                if len(circ.data) != len(circ2.data):
                    return False
                for gate, gate2 in zip(circ.data, circ2.data):
                    if gate != gate2:
                        return False
                    else:
                        for qubit, qubit2 in zip(gate[1], gate2[1]):
                            if qubit != qubit2:
                                return False
                return True

            self.assertEqual(test_equal_qiskit(circ, circ2), True)

    def test_cirq2pyquil(self):
        ref_qprog = pyquil.quil.Program().inst(
            pyquil.gates.X(0),
            pyquil.gates.T(1),
            pyquil.gates.CNOT(0, 1),
            pyquil.gates.SWAP(0, 1),
            pyquil.gates.CZ(1, 0),
        )

        qubits = [cirq.GridQubit(i, 0) for i in ref_qprog.get_qubits()]

        gates = [
            cirq.X(qubits[0]),
            cirq.T(qubits[1]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.SWAP(qubits[0], qubits[1]),
            cirq.CZ(qubits[1], qubits[0]),
        ]

        circuit = cirq.Circuit()
        circuit.append(gates, strategy=cirq.circuits.InsertStrategy.EARLIEST)

        qprog = cirq2pyquil(circuit)
        self.assertEqual(qprog, ref_qprog)

    def test_cirq2pyquil_RZ(self):
        circuit = cirq.Circuit()
        circuit.append(cirq.Rz(rads=0.1)(cirq.GridQubit(0, 0)))
        qprog = cirq2pyquil(circuit)
        self.assertEqual(qprog[0].name, "RZ")
        self.assertAlmostEqual(qprog[0].params[0], 0.1)

    def test_cirq2pyquil_PhasedX(self):
        circuit = cirq.Circuit()
        circuit.append(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5)(cirq.GridQubit(0, 0))
        )
        qprog = cirq2pyquil(circuit)
        self.assertEqual(qprog[0].name, "RZ")
        self.assertAlmostEqual(qprog[0].params[0], -np.pi / 4)
        self.assertEqual(qprog[1].name, "RX")
        self.assertAlmostEqual(qprog[1].params[0], np.pi / 2)
        self.assertEqual(qprog[2].name, "T")

    def test_to_quil(self):
        qubits = [Qubit(i) for i in range(0, 3)]
        gate_H0 = Gate("H", [qubits[0]])
        gate_CNOT01 = Gate("CNOT", [qubits[0], qubits[1]])
        gate_T2 = Gate("T", [qubits[2]])
        gate_CZ12 = Gate("CZ", [qubits[1], qubits[2]])

        circ1 = Circuit("circ1")
        circ1.qubits = qubits
        circ1.gates = [gate_H0, gate_CNOT01, gate_T2]

        self.assertEqual(circ1.to_quil(), circ1.to_pyquil().out())

    def test_to_text_diagram(self):
        qubits = [Qubit(i) for i in range(0, 3)]
        gate_H0 = Gate("H", [qubits[0]])
        gate_CNOT01 = Gate("CNOT", [qubits[0], qubits[1]])
        gate_T2 = Gate("T", [qubits[2]])
        gate_CZ12 = Gate("CZ", [qubits[1], qubits[2]])

        circ1 = Circuit("circ1")
        circ1.qubits = qubits
        circ1.gates = [gate_H0, gate_CNOT01, gate_T2]

        self.assertEqual(circ1.to_text_diagram(), circ1.to_cirq().to_text_diagram())

    def test_n_multiqubit_gates(self):
        qubits = [Qubit(i) for i in range(0, 3)]
        gate_H0 = Gate("H", [qubits[0]])
        gate_CNOT01 = Gate("CNOT", [qubits[0], qubits[1]])
        gate_T2 = Gate("T", [qubits[2]])
        gate_CZ12 = Gate("CZ", [qubits[1], qubits[2]])

        circuit = Circuit()
        circuit.qubits = qubits
        circuit.gates = [gate_H0, gate_CNOT01, gate_T2, gate_CZ12]
        self.assertEqual(circuit.n_multiqubit_gates, 2)

    def test_to_qpic(self):
        circuit = Circuit()
        circuit.qubits = [Qubit(i) for i in range(0, 3)]
        circuit.gates = []
        circuit.gates.append(Gate("H", [circuit.qubits[0]]))
        circuit.gates.append(Gate("CNOT", [circuit.qubits[0], circuit.qubits[1]]))
        circuit.gates.append(Gate("X", [circuit.qubits[2]]))
        circuit.gates.append(Gate("CZ", [circuit.qubits[2], circuit.qubits[1]]))
        circuit.gates.append(
            Gate("CPHASE", [circuit.qubits[2], circuit.qubits[1]], [0.1])
        )
        circuit.gates.append(Gate("ZXZ", [circuit.qubits[2]], [np.pi / 2, np.pi]))
        qpic_string = circuit.to_qpic()

        target_string = dedent(
            """w0 W 0
        w1 W 1
        w2 W 2
        w0 G {H} 
        w1 C w0
        w2 N
        w1 Z w2
        w2 w1 G {CPHASE(0.10)}  width=72
        w2 G {ZXZ(1.57, 3.14)}  width=90"""
        )

        for line_pair in zip(qpic_string.splitlines(), target_string.splitlines()):
            self.assertEqual(line_pair[0].strip(), line_pair[1].strip())

    def test_damping_gates(self):
        p = 0.1
        qubits = [Qubit(0), Qubit(1)]
        circuit = Circuit()
        circuit.qubits = qubits
        circuit.gates = [Gate("Z", [qubits[0]]), Gate("Da", [qubits[1]], [p])]
        target_matrix = np.diag([1.0, np.sqrt(1 - p), -1.0, -np.sqrt(1 - p)])
        self.assertTrue(np.allclose(circuit.to_unitary(), target_matrix))

        circuit = Circuit()
        circuit.qubits = qubits
        circuit.gates = [Gate("Z", [qubits[0]]), Gate("Db", [qubits[1]], [p])]
        target_matrix = np.zeros((4, 4))
        target_matrix[0, 1] = np.sqrt(p)
        target_matrix[2, 3] = -np.sqrt(p)
        print(target_matrix)
        print(circuit.to_unitary())
        self.assertTrue(np.allclose(circuit.to_unitary(), target_matrix))

    def test_retain_qubit_index_to_and_from_cirq(self):
        circ = create_random_circuit(10, 20, seed=RNDSEED)
        while circ.gates[0].qubits[0].index == 0:
            circ = create_random_circuit(10, 20, seed=RNDSEED)

        zirc = Circuit(circ.to_cirq())

        self.assertTrue(compare_unitary(circ.to_unitary(), zirc.to_unitary()))

    def test_retain_qubit_index_to_and_from_qiskit(self):
        circ = create_random_circuit(10, 20, seed=RNDSEED)
        while circ.gates[0].qubits[0].index == 0:
            circ = create_random_circuit(10, 20, seed=RNDSEED)

        zirc = Circuit(circ.to_qiskit())

        self.assertTrue(compare_unitary(circ.to_unitary(), zirc.to_unitary()))

    def test_save_circuit_from_qiskit(self):
        circ = create_random_circuit(11, 100, seed=RNDSEED)
        circ = Circuit(circ.to_qiskit())
        save_circuit(circ, "circuit_from_qiskit.json")
        os.remove("circuit_from_qiskit.json")

    def test_save_circuit_from_qiskit_two_circuits(self):
        circ1 = create_random_circuit(11, 100, seed=RNDSEED)
        circ2 = create_random_circuit(11, 100, seed=RNDSEED + 1)
        # Create two circuits that came from qiskit and add together
        circ1 = Circuit(circ1.to_qiskit())
        circ1 += Circuit(circ2.to_qiskit())
        # Translate new circuit (two qregs) to qiskit and back
        circ = Circuit(circ1.to_qiskit())
        save_circuit(circ, "circuit_from_qiskit.json")
        os.remove("circuit_from_qiskit.json")

    def test_exp_S_inverse(self):
        # Test the corner case of converting a ZPowGate with exponent -0.5 from
        # cirq. This gate will be represented by the string 'S**-1'.
        gate = cirq.S(cirq.LineQubit(0)) ** -1
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(gate)
        zircuit = Circuit(cirq_circuit)
        self.assertTrue(
            compare_unitary(zircuit.to_unitary(), cirq.unitary(cirq_circuit))
        )


if __name__ == "__main__":
    unittest.main()
