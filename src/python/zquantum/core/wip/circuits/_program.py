from zquantum.core.circuits._circuit import Circuit

from ._basis_gateset import BasisGateset


class Program:
    def __init__(self):
        self.instructions = []

    def append(self, circuit: Circuit, gateset: BasisGateset):
        """Append a circuit to the program in the basis gateset.

        Args:
            circuit (Circuit): Circuit to be appended to the program.
            gateset (BasisGateset): Gateset from which we want the appended
            circuit to be composed of.
        """
        self.instructions.append((circuit, gateset))

    def decompose(self):
        """Decompose operations in the program into their respective
        basis gatesets.
        """
        for index, instruction in enumerate(self.instructions):
            circuit, gateset = instruction
            decomposed_circuit = gateset.decompose_circuit(circuit)
            self.instructions[index] = (decomposed_circuit, gateset)
