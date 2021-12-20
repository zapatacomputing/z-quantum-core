from _basis_gateset import BasisGateset
from _circuit import Circuit


class Program:
    def __init__(self):
        self.instructions = []

    def append(self, circuit: Circuit, gateset: BasisGateset):
        self.instructions.append((circuit, gateset))

    def decompose(self):
        for index, instruction in enumerate(self.instructions):
            circuit, gateset = instruction
            decomposed_circuit = gateset.decompose_circuit(circuit)
            self.instructions[index] = decomposed_circuit
