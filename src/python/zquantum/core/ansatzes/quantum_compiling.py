from ..interfaces.ansatz import Ansatz
from ..circuit import Circuit, Qubit, Gate
from ..interfaces.ansatz_utils import ansatz_property

from typing import Optional, List
from overrides import overrides
import numpy as np
import sympy


class HEAQuantumCompilingAnsatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")

    def __init__(self, number_of_layers: int, number_of_qubits: int):
        """
        An ansatz implementation used for running the Quantum Compiling Algorithm

        Args:
            number_of_layers (int): number of layers in the circuit.
            number_of_qubits (int): number of qubits in the circuit.

        Attributes:
            number_of_qubits (int): See Args
            number_of_layers (int): See Args
        """
        super().__init__(number_of_layers)
        assert number_of_qubits % 2 == 0
        self._number_of_qubits = number_of_qubits

    def _build_rotational_subcircuit(self, circuit, parameters) -> Circuit:

        # Add Rz(theta) Rx(pi/2) Rz(theta') Rx(pi/2) Rz(theta'')
        for qubit_index, qubit in enumerate(circuit.qubits):

            qubit_parameters = parameters[qubit_index * 3 : (qubit_index + 1) * 3]

            circuit.gates.append(Gate("Rz", [qubit], [qubit_parameters[0]]))
            circuit.gates.append(Gate("Rx", [qubit], [np.pi / 2]))
            circuit.gates.append(Gate("Rz", [qubit], [qubit_parameters[1]]))
            circuit.gates.append(Gate("Rx", [qubit], [np.pi / 2]))
            circuit.gates.append(Gate("Rz", [qubit], [qubit_parameters[2]]))

        return circuit

    def _build_circuit_layer(self, parameters: np.ndarray) -> Circuit:
        circuit_layer = Circuit()
        circuit_layer.qubits = [Qubit(i) for i in range(self.number_of_qubits)]

        # Add Rz(theta) Rx(pi/2) Rz(theta') Rx(pi/2) Rz(theta'')
        circuit_layer = self._build_rotational_subcircuit(
            circuit_layer, parameters[: 3 * self.number_of_qubits]
        )

        # Add CNOT(x, x+1) for x in even(qubits)
        for control, target in zip(
            circuit_layer.qubits[::2], circuit_layer.qubits[1::2]
        ):  # loop over qubits 0, 2, 4...
            circuit_layer.gates.append(Gate("CNOT", [control, target], []))

        # Add Rz(theta) Rx(pi/2) Rz(theta') Rx(pi/2) Rz(theta'')
        circuit_layer = self._build_rotational_subcircuit(
            circuit_layer,
            parameters[3 * self.number_of_qubits : 6 * self.number_of_qubits],
        )

        # Add CNOT layer working "inside -> out", skipping every other qubit
        for qubit_index, qubit in enumerate(
            circuit_layer.qubits[: int(self.number_of_qubits / 2)][::-1][::2]
        ):
            control = qubit
            target = circuit_layer.qubits[self.number_of_qubits - qubit.index - 1]
            circuit_layer.gates.append(Gate("CNOT", [control, target], []))

            if not (qubit.index == 0 and self.number_of_qubits % 4 != 0):
                control = circuit_layer.qubits[self.number_of_qubits - qubit.index]
                target = circuit_layer.qubits[qubit.index - 1]
                circuit_layer.gates.append(Gate("CNOT", [control, target], []))

        return circuit_layer

    @overrides
    def _generate_circuit(self, parameters: Optional[np.ndarray] = None) -> Circuit:
        """Builds the ansatz circuit (based on: 2011.12245, Fig. 1)

        Args:
            params (numpy.array): input parameters of the circuit (1d array).

        Returns:
            Circuit
        """
        if parameters is None:
            parameters = self.symbols

        assert len(parameters) == self.number_of_params

        circuit = Circuit()
        for layer_index in range(self.number_of_layers):
            circuit += self._build_circuit_layer(
                parameters[
                    layer_index
                    * self.number_of_params_per_layer : (layer_index + 1)
                    * self.number_of_params_per_layer
                ]
            )
        return circuit

    @property
    def number_of_params(self) -> int:
        """
        Returns number of parameters in the ansatz.
        """
        return self.number_of_params_per_layer * self.number_of_layers

    @property
    def number_of_params_per_layer(self) -> int:
        """
        Returns number of parameters in the ansatz.
        """
        return 3 * self.number_of_qubits * 2

    @property
    def symbols(self) -> List[sympy.Symbol]:
        """
        Returns a list of symbolic parameters used for creating the ansatz.
        The order of the symbols should match the order in which parameters should be passed for creating executable circuit.
        """
        return np.asarray(
            [sympy.Symbol("theta_{}".format(i)) for i in range(self.number_of_params)]
        )