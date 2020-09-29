import unittest
from ..testing import create_random_circuit
from .william_compile import compile
import copy
from qeqiskit.simulator import QiskitSimulator

from pyquil import Program
from pyquil.gates import *
from ..circuit import Circuit
import numpy as np


class TestWilliamCompile(unittest.TestCase):
    def test_basic_circuit(self):
        simulator = QiskitSimulator("statevector_simulator")
        circuit = Circuit(
            Program(RX(0.159 * np.pi, 0), H(1), H(1), RZ(0.075 * np.pi, 1))
        )
        new_circuit = Circuit(Program(RX(0.159 * np.pi, 0), RZ(0.075 * np.pi, 1)))

        wavefunction = simulator.get_wavefunction(circuit)
        new_wavefunction = simulator.get_wavefunction(new_circuit)

        for probabilities, new_probabilities in zip(
            wavefunction.probabilities(), new_wavefunction.probabilities()
        ):
            self.assertAlmostEqual(probabilities, new_probabilities)

        circuit = Circuit(Program(Z(0), RZ(-0.237 * np.pi, 1), H(1), H(1)))
        new_circuit = Circuit(Program(Z(0), RZ(-0.237 * np.pi, 1)))

        wavefunction = simulator.get_wavefunction(circuit)
        new_wavefunction = simulator.get_wavefunction(new_circuit)

        for probabilities, new_probabilities in zip(
            wavefunction.probabilities(), new_wavefunction.probabilities()
        ):
            self.assertAlmostEqual(probabilities, new_probabilities)

    def test_compiler(self):
        num_trials = 100
        possible_number_of_qubits = [2, 3, 4, 5, 6]
        possible_number_of_gates = [2, 4, 8, 16, 32, 64]
        simulator = QiskitSimulator("statevector_simulator")

        for number_of_qubits in possible_number_of_qubits:
            for number_of_gates in possible_number_of_gates:
                for _ in range(num_trials):
                    print()
                    circuit = create_random_circuit(number_of_qubits, number_of_gates)
                    print(
                        [
                            (
                                gate.name,
                                [qubit.index for qubit in gate.qubits],
                                [param for param in gate.params],
                            )
                            for gate in circuit.gates
                        ]
                    )

                    new_circuit = copy.deepcopy(circuit)
                    new_circuit.gates = compile(circuit.gates)
                    print(
                        [
                            (
                                gate.name,
                                [qubit.index for qubit in gate.qubits],
                                [param for param in gate.params],
                            )
                            for gate in new_circuit.gates
                        ]
                    )

                    if len(circuit.gates) > len(new_circuit.gates):
                        print("Reduced gate size")

                    wavefunction = simulator.get_wavefunction(circuit)
                    new_wavefunction = simulator.get_wavefunction(new_circuit)

                    for probabilities, new_probabilities in zip(
                        wavefunction.probabilities(), new_wavefunction.probabilities()
                    ):
                        self.assertAlmostEqual(probabilities, new_probabilities)

