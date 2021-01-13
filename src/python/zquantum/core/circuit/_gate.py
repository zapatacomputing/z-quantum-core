"""Base class for quantum gates"""

import sys
import numpy as np
import sympy
import copy

import cirq
import pyquil
import qiskit
from math import pi, exp

from pyquil.quilatom import quil_sin, quil_cos
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.quantumregister import Qubit as QiskitQubit
from qiskit.circuit.classicalregister import Clbit as QiskitClbit
from qiskit.circuit import ParameterExpression
from qiskit.aqua.circuits.gates import mct

from ._gateset import ALL_GATES
from ._qubit import Qubit


class Gate(object):
    """Class for storing information associated with a quantum gate.

    Attributes:
        name: string
            Name of the gate. Examples include 'H', 'CNOT' etc.
            The validity of the name goes hand in hand with the specific package
            for quantum computation, as described in 'label'.
        qubits: list[Qubit]
            List of core.qubit.Qubit object that the gate operates on.
        params: list[any]
            List of parameters associated with the gate.
        The above three attributes follow a standard convention that is enforced throughout
        z-quantum-core

        (optional attribute)
        info: dictionary
            Additional information on the gate. Entries:
            label: string
                Name of the package that natively generated the gate object.
                Currently supported:
                    pyquil
                    cirq
                    qiskit
            object: any
                The gate object in the native package that describes the quantum gate, such
                as pyquil Program object or cirq GateOperation object.
    """

    def __init__(
        self,
        name="none",
        qubits=None,
        params=None,
        control_qubits=None,
        target_qubits=None,
        all_circuit_qubits=None,
    ):

        if qubits is None:
            qubits = []
        if params is None:
            params = []

        self.name = name
        self.qubits = qubits
        self.params = params
        self.all_circuit_qubits = all_circuit_qubits
        self.control_qubits = control_qubits
        self.target_qubits = target_qubits

        # optional attributes
        self.info = {"label": "none"}

    def __str__(self):
        return str(self.to_cirq())

    def __eq__(self, another_gate):
        if self.name != another_gate.name:
            return False
        if len(self.qubits) != len(another_gate.qubits):
            return False
        for i in range(len(self.qubits)):
            if str(self.qubits[i]) != str(another_gate.qubits[i]):
                return False
        if len(self.params) != len(another_gate.params):
            return False
        for i in range(len(self.params)):
            if self.params[i] != another_gate.params[i]:
                if sympy.simplify(self.params[i] - another_gate.params[i]) != 0:
                    return False
        return True

    def __repr__(self):
        return f"zquantum.core.circuit.Gate(name={self.name}, qubits={self.qubits}, params={self.params})"

    def evaluate(self, symbols_map):
        """
        Returns a copy of a gate with specified symbolic parameters evaluated to provided values.

        Args:
            symbols_map list(tuple(sympy.Basic, number)): List containing symbols and values that they should take.
        """
        gate_class = type(self)
        evaluated_gate = gate_class(
            name=self.name, qubits=self.qubits, params=copy.copy(self.params)
        )

        for i in range(len(evaluated_gate.params)):
            if isinstance(evaluated_gate.params[i], sympy.Basic):
                number = evaluated_gate.params[i].subs(symbols_map).evalf()
                if isinstance(number, sympy.Number):
                    number = float(number)
                evaluated_gate.params[i] = number
        return evaluated_gate

    @property
    def symbolic_params(self):
        """
        Returns a list of symbolic parameters used in the gate

        Returns:
            list: list containing all the sympy symbols used in gate params
        """
        all_symbols = []
        for param in self.params:
            if isinstance(param, sympy.Basic):
                for symbol in param.free_symbols:
                    all_symbols.append(symbol)

        unique_symbols = sorted(list(set(all_symbols)), key=str)
        return unique_symbols

    def get_param_string(self):
        r"""Get a string containing the parameters, e.g.

        (0.2, $\pi$/3)

        Returns:
            str: a string representing the parameters
        """
        if len(self.params) == 0:
            return ""

        param_string = "("
        for i in range(len(self.params) - 1):
            param_string += "{:.2f}, ".format(self.params[i])

        param_string += "{:.2f})".format(self.params[-1])

        return param_string

    def to_dict(self, serialize_params=False):
        """Convert the gate back to a dictionary.
        Params might be sympy objects, which are not serializable by default.
        In order to convert them to strings, `serialize_params` must be set to true.
        If all the params are numerical, they will be serializable anyway.

        Args:
            serialize_params(bool): flag indicating whether sympy parameters should be serialized.

        Returns:
            A dictionary with only serializable values.
        """
        if serialize_params:
            params = []
            for param in self.params:
                if isinstance(param, sympy.Basic):
                    params.append(str(param))
                else:
                    params.append(param)
        else:
            params = self.params
        return {
            "name": self.name,
            "qubits": [qubit.to_dict() for qubit in self.qubits],
            "info": self.info,
            "params": params,
        }

    def to_unitary(self):
        """Returns a unitary matrix representing the gate."""
        if self.name in {
            "I",
            "X",
            "Y",
            "Z",
            "H",
            "S",
            "T",
            "PHASE",
            "Rx",
            "Ry",
            "Rz",
            "ZXZ",
            "RH",
            "CNOT",
            "CZ",
            "CPHASE",
            "SWAP",
            "XX",
            "YY",
            "ZZ",
        }:  # gates defined in cirq
            return self.to_cirq().gate._unitary_()
        else:
            raise NotImplementedError(
                "Gate {} currently not supported.".format(self.name)
            )

    @classmethod
    def from_dict(cls, dict):
        """Generates Gate object from dictionary. This is the inverse operation to the
        serialization function to_dict.
        If any of the gate params is string it will be converted to sympy symbol.

        dict: dictionary
        Contains information needed to specify the Gate. See to_dict for details.

        Return:
        A core.gate.Gate object.
        """
        params = []
        for param in dict["params"]:
            if type(param) is str:
                params.append(sympy.sympify(param))
            else:
                params.append(param)
        output = cls(
            dict["name"],
            [Qubit.from_dict(qubit) for qubit in dict["qubits"]],
            params,
        )
        output.info = dict["info"]
        return output

    def to_pyquil(self):
        """Converts the current gate to a pyquil gate.

        Return:
        A pyquil Program object that corresponds to the specification of QGate.
        """

        if self.name not in ALL_GATES:
            sys.exit(f"Gate {self.name} currently not supported.")

        q1 = self.qubits[0].index
        if len(self.qubits) >= 2:
            q2 = self.qubits[1].index
        if len(self.qubits) >= 3:
            q3 = self.qubits[2].index
        if len(self.params) > 0:
            params = self.params

        # single-qubit gates
        if self.name == "I":  # identity
            return pyquil.gates.I(q1)
        if self.name == "X":  # Pauli X
            return pyquil.gates.X(q1)
        if self.name == "Y":  # Pauli Y
            return pyquil.gates.Y(q1)
        if self.name == "Z":  # Pauli Z
            return pyquil.gates.Z(q1)
        if self.name == "H":  # Hadamard
            return pyquil.gates.H(q1)
        if self.name == "S":  # S gate
            return pyquil.gates.S(q1)
        if self.name == "T":  # T gate
            return pyquil.gates.T(q1)
        if self.name == "Rx":  # Single-qubit X rotation
            return pyquil.gates.RX(params[0], q1)
        if self.name == "Ry":  # Single-qubit Y rotation
            return pyquil.gates.RY(params[0], q1)
        if self.name == "Rz":  # Single-qubit Z rotation
            return pyquil.gates.RZ(params[0], q1)
        if self.name == "PHASE":  # Phase gate
            return pyquil.gates.PHASE(params[0], q1)

        # two-qubit gates
        if self.name == "CNOT":
            return pyquil.gates.CNOT(q1, q2)
        if self.name == "CZ":
            return pyquil.gates.CZ(q1, q2)
        if self.name == "CPHASE":
            return pyquil.gates.CPHASE(params[0], q1, q2)
        if self.name == "SWAP":
            return pyquil.gates.SWAP(q1, q2)

    def to_cirq(self, input_cirq_qubits=None):
        """Convert to a cirq gate.

        Args:
            input_cirq_qubits: list[cirq.LineQubit]
                (optional) a list of cirq Qubits that the gate can act on. If not provided
                the function will generate new cirq.LineQubit objects.
        Returns:
        A cirq Circuit object that corresponds to the specification of the quantum gate.
            In the special case the gate itself was natively generated from cirq, the function
            will faithfully reproduce the original GateOperation object, taking into account
            whether the gate acts on GridQubit objects or LineQubit objects.
            In the other cases the resulting cirq gate simply assumes that the qubits are
            LineQubit objects.
        """

        if self.name not in ALL_GATES:
            sys.exit("Gate {} currently not supported.".format(self.name))

        q_inds = []
        q_inds.append(self.qubits[0].index)
        if len(self.qubits) >= 2:
            q_inds.append(self.qubits[1].index)
        if len(self.qubits) >= 3:
            q_inds.append(self.qubits[2].index)

        cirq_qubits = []
        if input_cirq_qubits == None:
            for q in self.qubits:
                if q.info["label"] == "cirq":
                    qkey = q.info["QubitKey"]
                    if q.info["QubitType"] == "GridQubit":
                        cirq_qubits.append(cirq.GridQubit(qkey[0], qkey[1]))
                    if q.info["QubitType"] == "LineQubit":
                        cirq_qubits.append(cirq.LineQubit(qkey))
                else:
                    cirq_qubits.append(cirq.LineQubit(q.index))
        else:
            cirq_qubits = [input_cirq_qubits[x] for x in [q.index for q in self.qubits]]

        if len(self.params) > 0:
            params = self.params

        # single-qubit gates
        if self.name == "I":  # Identity
            return cirq.I(cirq_qubits[0])
        if self.name == "X":  # Pauli X
            return cirq.X(cirq_qubits[0])
        if self.name == "Y":  # Pauli Y
            return cirq.Y(cirq_qubits[0])
        if self.name == "Z":  # Pauli Z
            return cirq.Z(cirq_qubits[0])
        if self.name == "H":  # Hadamard
            return cirq.H(cirq_qubits[0])
        if self.name == "S":  # S gate
            return cirq.S(cirq_qubits[0])
        if self.name == "T":  # T gate
            return cirq.T(cirq_qubits[0])
        if self.name == "Rx":  # Single-qubit X rotation
            return cirq.rx(params[0])(cirq_qubits[0])
        if self.name == "Ry":  # Single-qubit Y rotation
            return cirq.ry(params[0])(cirq_qubits[0])
        if self.name == "Rz":  # Single-qubit Z rotation
            return cirq.rz(params[0])(cirq_qubits[0])
        if self.name == "PHASE":  # Phase gate
            return cirq.Z(cirq_qubits[0]) ** (params[0] / pi)
        if self.name == "ZXZ":  # PhasedXPowGate gate
            g = cirq.PhasedXPowGate(
                phase_exponent=params[0] / pi, exponent=params[1] / pi
            )
            return g(cirq_qubits[0])
        if self.name == "RH":  # HPowGate
            g = cirq.H ** (params[0] / pi)
            return g(cirq_qubits[0])
        if self.name == "Da":  # Damping alpha gate
            g = DampingAlpha(params[0])
            return g(cirq_qubits[0])
        if self.name == "Db":  # Damping beta gate
            g = DampingBeta(params[0])
            return g(cirq_qubits[0])

        # two-qubit gates
        if self.name == "CNOT":
            return cirq.CNOT(cirq_qubits[0], cirq_qubits[1])
        if self.name == "CZ":
            return cirq.CZ(cirq_qubits[0], cirq_qubits[1])
        if self.name == "CPHASE":
            return cirq.CZPowGate(exponent=params[0] / pi)(
                cirq_qubits[0], cirq_qubits[1]
            )
        if self.name == "SWAP":
            return cirq.SWAP(cirq_qubits[0], cirq_qubits[1])
        if self.name == "XX":
            return cirq.XX(cirq_qubits[0], cirq_qubits[1]) ** (params[0] * 2 / pi)
        if self.name == "YY":
            return cirq.YY(cirq_qubits[0], cirq_qubits[1]) ** (params[0] * 2 / pi)
        if self.name == "ZZ":
            return cirq.ZZ(cirq_qubits[0], cirq_qubits[1]) ** (params[0] * 2 / pi)

    def to_qiskit(self, qreg, creg):
        """Converts a Gate object to a qiskit object.

        Args:
            qreg: QuantumRegister
                Optional feature in case the original circuit is not contructed from qiskit.
                Then we will use a single QuantumRegister for all of the qubits for the qiskit
                QuantumCircuit object.
        Returns:
            A list of length N*3 where N is the number of gates used in the
            decomposition. For each gate, the items appended to the list are,
            in order, the qiskit gate object, the qubits involved in the gate
            (described by a tuple of the quantum register and the index), and
            lastly, the classical register (for now the classical register is
            always empty, except for MEASURE)
        """

        if self.name not in ALL_GATES:
            sys.exit("Gate currently not supported.")

        qiskit_qubits = []
        qiskit_bits = []
        for q in self.qubits:
            qiskit_qubits.append(QiskitQubit(qreg, q.index))
            qiskit_bits.append(QiskitClbit(creg, q.index))

        if len(self.params) > 0:
            params = copy.copy(self.params)
            for i in range(len(params)):
                if isinstance(params[i], sympy.Basic):
                    params[i] = ParameterExpression({}, params[i])
        # single-qubit gates
        if self.name == "I":
            return [qiskit.extensions.standard.IGate(), [qiskit_qubits[0]], []]
        if self.name == "X":
            return [qiskit.extensions.standard.XGate(), [qiskit_qubits[0]], []]
        if self.name == "Y":
            return [qiskit.extensions.standard.YGate(), [qiskit_qubits[0]], []]
        if self.name == "Z":
            return [qiskit.extensions.standard.ZGate(), [qiskit_qubits[0]], []]
        if self.name == "H":
            return [qiskit.extensions.standard.HGate(), [qiskit_qubits[0]], []]
        if self.name == "T":
            return [qiskit.extensions.standard.TGate(), [qiskit_qubits[0]], []]
        if self.name == "S":
            return [qiskit.extensions.standard.SGate(), [qiskit_qubits[0]], []]
        if self.name == "Rx":
            return [
                qiskit.extensions.standard.RXGate(params[0]),
                [qiskit_qubits[0]],
                [],
            ]
        if self.name == "Ry":
            return [
                qiskit.extensions.standard.RYGate(params[0]),
                [qiskit_qubits[0]],
                [],
            ]
        if self.name == "Rz" or self.name == "PHASE":
            return [
                qiskit.extensions.standard.RZGate(params[0]),
                [qiskit_qubits[0]],
                [],
            ]
        if self.name == "ZXZ":  # PhasedXPowGate gate (from cirq)
            # Hard-coded decomposition is used for now.
            return [
                qiskit.extensions.standard.RXGate(-params[0]),
                [qiskit_qubits[0]],
                [],
                qiskit.extensions.standard.RZGate(params[1]),
                [qiskit_qubits[0]],
                [],
                qiskit.extensions.standard.RXGate(params[0]),
                [qiskit_qubits[0]],
                [],
            ]
        if self.name == "RH":  # HPowGate (from cirq)
            # Hard-coded decomposition is used for now.
            return [
                qiskit.extensions.standard.RYGate(pi / 4),
                [qiskit_qubits[0]],
                [],
                qiskit.extensions.standard.RZGate(params[0]),
                [qiskit_qubits[0]],
                [],
                qiskit.extensions.standard.RYGate(-pi / 4),
                [qiskit_qubits[0]],
                [],
            ]

        # two-qubit gates
        if self.name == "CNOT":
            return [
                qiskit.extensions.standard.CnotGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
            ]
        if self.name == "CZ":
            return [
                qiskit.extensions.standard.CzGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
            ]
        if self.name == "CPHASE":
            return [
                qiskit.extensions.standard.RXGate(pi / 2),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.RYGate(pi - params[0] / 2),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.CzGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.RYGate(-(pi - params[0] / 2)),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.RXGate(-pi),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.CzGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.RXGate(pi / 2),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.RZGate(params[0] / 2),
                [qiskit_qubits[0]],
                [],
            ]
        if self.name == "SWAP":
            return [
                qiskit.extensions.standard.SwapGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
            ]
        if self.name == "XX":
            # Hard-coded decomposition is used for now. The compilation is inspired by the approach described in arXiv:1001.3855 [quant-ph]
            return [
                qiskit.extensions.standard.HGate(),
                [qiskit_qubits[0]],
                [],
                qiskit.extensions.standard.HGate(),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.CnotGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.RZGate(params[0] * 2),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.CnotGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.HGate(),
                [qiskit_qubits[0]],
                [],
                qiskit.extensions.standard.HGate(),
                [qiskit_qubits[1]],
                [],
            ]
        if self.name == "YY":
            # Hard-coded decomposition is used for now. The compilation is inspired by the approach described in arXiv:1001.3855 [quant-ph]
            return [
                qiskit.extensions.standard.SGate(),
                [qiskit_qubits[0]],
                [],
                qiskit.extensions.standard.SGate(),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.HGate(),
                [qiskit_qubits[0]],
                [],
                qiskit.extensions.standard.HGate(),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.CnotGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.RZGate(params[0] * 2),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.CnotGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.HGate(),
                [qiskit_qubits[0]],
                [],
                qiskit.extensions.standard.HGate(),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.SdgGate(),
                [qiskit_qubits[0]],
                [],
                qiskit.extensions.standard.SdgGate(),
                [qiskit_qubits[1]],
                [],
            ]
        if self.name == "ZZ":
            # Hard-coded decomposition is used for now. The compilation is inspired by the approach described in arXiv:1001.3855 [quant-ph]
            return [
                qiskit.extensions.standard.CnotGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.RZGate(params[0] * 2),
                [qiskit_qubits[1]],
                [],
                qiskit.extensions.standard.CnotGate(),
                [qiskit_qubits[0], qiskit_qubits[1]],
                [],
            ]
        if self.name == "CCX":
            return [
                qiskit.extensions.standard.ToffoliGate(),
                [qiskit_qubits[0], qiskit_qubits[1], qiskit_qubits[2]],
                [],
            ]
        if self.name == "MCT":
            gate = MCTGate(
                self.all_circuit_qubits,
                control_qubits=self.control_qubits,
                target_qubits=self.target_qubits,
            )
            return gate.qiskit_data
        if self.name == "MCU1":
            gate = PhaseOracle(
                self.params[0],
                self.all_circuit_qubits,
                control_qubits=self.control_qubits,
                target_qubits=self.target_qubits,
            )
            return gate.qiskit_data
        if self.name == "MCRY":
            gate = MCRY(
                self.params[0],
                self.all_circuit_qubits,
                control_qubits=self.control_qubits,
                target_qubits=self.target_qubits,
            )
            return gate.qiskit_data
        if self.name == "MEASURE":
            return [qiskit.circuit.measure.Measure(), qiskit_qubits, qiskit_bits]
        if self.name == "BARRIER":
            return [
                qiskit.extensions.standard.Barrier(len(qiskit_qubits)),
                qiskit_qubits,
                [],
            ]

    def to_qpic(self):
        """Get a string containing a qpic command for drawing the circuit.

        Note that the qpic wire corresponding to qubit 0 is assumed to be
        labeled w0, the wire corresponding to qubit 1 labeled w1, etc.

        Returns:
            str: string containing the qpic command
        """

        # Gates that have special symbols
        if self.name == "X":
            return "w{} N".format(self.qubits[0].index)
        elif self.name == "CNOT":
            # In qpic CNOT is target C control, in circuits it is CNOT(control, target)
            return "w{} C w{}".format(self.qubits[1].index, self.qubits[0].index)
        elif self.name == "SWAP":
            return "w{} w{} SWAP".format(self.qubits[0].index, self.qubits[1].index)
        elif self.name == "CZ":
            # In qpic CZ is target Z control, in circuits it is CZ(control, target)
            return "w{} Z w{}".format(self.qubits[1].index, self.qubits[0].index)

        # Generic gate
        else:

            # Which qubits the gate acts on
            qpic_string = ""
            for qubit in self.qubits:
                qpic_string += "w{} ".format(qubit.index)

            # The gate label. Note the extra curly braces so that whitespaces are
            # included in the gate label
            qpic_string += "G {{{}{}}} ".format(self.name, self.get_param_string())

            # Try to adjust the width so that the label fits
            if len(self.params) > 0:
                qpic_string += " width={}".format(
                    6 * len(self.name + self.get_param_string())
                )

            return qpic_string

    @classmethod
    def from_pyquil(cls, pyquil_gate, Qubit_list):
        """Generates a Gate object from a pyquil Gate object.

        Args:
            pyquil_gate: pyquil.quilbase.Gate
                The input pyquil Gate object.
            Qubit_list: list[Qubit]
                A list of core.Qubit objects.

        Returns:
            A core.Gate object.
        """

        output = cls()
        if pyquil_gate.name not in {"RX", "RY", "RZ"}:
            output.name = pyquil_gate.name
        else:
            if pyquil_gate.name == "RX":
                output.name = "Rx"
            if pyquil_gate.name == "RY":
                output.name = "Ry"
            if pyquil_gate.name == "RZ":
                output.name = "Rz"
        output.qubits = Qubit_list
        output.info["label"] = "pyquil"
        output.params = pyquil_gate.params
        return output

    @classmethod
    def from_cirq(cls, cirq_gate, Qubit_list):
        """Generates a Gate object from a cirq GateOperation.

        Args:
            cirq_gate: Cirq.ops.GateOperation
                The input cirq GateOperation object.
            Qubit_list: list[Qubit]
                A list of core.Qubit objects.

        Returns:
            A core.Gate object. Here the 'params' entry stores the 'exponent' of
            the GateOperation object from cirq. The 'exponent' entry is crucial because
            in cirq implementation, two gates

            Z**1.0
            Z**0.75

            both have their name string to be 'Z' while the first one is the Z gate, while the
            second one is Z rotation by angle 2·0.75π = 1.5π. This is in contrast with pyquil
            where {X,Y,Z} are fixed single-qubit gates that are different from the rotation
            gates {RX, RY, RZ}.
        """

        output = cls()
        gate_string = str(cirq_gate.gate)
        if (
            len(gate_string.split("(")) > 1
        ):  # test if the gate string contains parentheses
            parsed_gatename = gate_string.split("(")[0]
            if parsed_gatename == "PhasedX" or parsed_gatename == "PhX":
                output.name = "ZXZ"
                output.params = [
                    cirq_gate.gate.phase_exponent * pi,
                    cirq_gate.gate.exponent * pi,
                ]
            elif parsed_gatename == "Rx":
                output.name = "Rx"
                output.params = [cirq_gate.gate.exponent * pi]
            elif parsed_gatename == "Ry":
                output.name = "Ry"
                output.params = [cirq_gate.gate.exponent * pi]
            elif parsed_gatename == "Rz":
                output.name = "Rz"
                output.params = [cirq_gate.gate.exponent * pi]
            else:
                raise NotImplementedError(
                    "The cirq gate {} is currently not supported".format(
                        parsed_gatename
                    )
                )
        else:
            parsed_gatename = (
                str(cirq_gate.gate).replace("**", " ").replace("^", " ").split()
            )
            name_str = parsed_gatename[0]
            if name_str in {
                "I",
                "X",
                "Y",
                "Z",
                "H",
                "T",
                "S",
                "CNOT",
                "SWAP",
                "CZ",
                "XX",
                "YY",
                "ZZ",
            }:
                if len(parsed_gatename) == 1:  # discrete gate
                    output.name = name_str
                    if name_str in {"XX", "YY", "ZZ"}:
                        output.params = [pi / 2]
                else:  # rotation with respect to axes
                    if name_str == "X":
                        output.name = "Rx"
                        output.params = [cirq_gate.gate.exponent * pi]
                    elif name_str == "I":
                        output.name = "I"
                        output.params = []
                    elif name_str == "Y":
                        output.name = "Ry"
                        output.params = [cirq_gate.gate.exponent * pi]
                    elif name_str == "Z":
                        output.name = "PHASE"
                        output.params = [cirq_gate.gate.exponent * pi]
                    elif name_str == "H":
                        output.name = "RH"
                        output.params = [cirq_gate.gate.exponent * pi]
                    elif name_str == "CZ":
                        output.name = "CPHASE"
                        output.params = [cirq_gate.gate.exponent * pi]
                    elif name_str == "XX":
                        output.name = "XX"
                        output.params = [cirq_gate.gate.exponent * pi / 2]
                    elif name_str == "YY":
                        output.name = "YY"
                        output.params = [cirq_gate.gate.exponent * pi / 2]
                    elif name_str == "ZZ":
                        output.name = "ZZ"
                        output.params = [cirq_gate.gate.exponent * pi / 2]
                    elif name_str == "S":
                        # In cirq, ZPowGate(exponent=-0.5) has a string
                        # representation of 'S**-1'.
                        output.name = "PHASE"
                        output.params = [cirq_gate.gate.exponent * pi]
                    else:
                        raise NotImplementedError(
                            "The cirq gate {} is currently not supported in exponential format".format(
                                name_str
                            )
                        )
            else:
                raise NotImplementedError(
                    "The cirq gate {} is currently not supported".format(name_str)
                )

        output.qubits = Qubit_list
        output.info["label"] = "cirq"

        return output

    @classmethod
    def from_qiskit(cls, qiskit_gate, Qubit_list):
        """Generates a Gate object from a qiskit Gate object.

        Args:
            qiskit_gate: qiskit.circuit.Gate
                The input qiskit Gate object.
            Qubit_list: list[Qubit]
                A list of core.Qubit objects.

        """

        output = cls()
        if qiskit_gate.name in {"x", "y", "z", "h", "t", "s"}:
            output.name = qiskit_gate.name.upper()
        elif qiskit_gate.name in {"id"}:
            output.name = "I"
        elif qiskit_gate.name in {"rx", "ry", "rz"}:
            output.name = "R" + qiskit_gate.name[1]
        elif qiskit_gate.name == "cx":
            output.name = "CNOT"
        elif qiskit_gate.name in {"cz", "swap"}:
            output.name = qiskit_gate.name.upper()
        elif qiskit_gate.name in {"measure", "barrier"}:
            output.name = qiskit_gate.name.upper()
        elif qiskit_gate.name == "ccx":
            output.name = "CCX"
        elif qiskit_gate.name == "mct":
            output.name = "MCT"
        elif qiskit_gate.name == "mcu1":
            output.name = "MCU1"
        elif qiskit_gate.name == "u1":
            output.name = "PHASE"
        else:
            raise NotImplementedError(
                "The gate {} is currently not supported.".format(qiskit_gate.name)
            )

        output.qubits = Qubit_list
        if len(qiskit_gate.params) > 0:
            output.params = []
            for x in qiskit_gate.params:
                if isinstance(x, ParameterExpression):
                    output.params.append(x._symbol_expr)
                else:
                    output.params.append(float(x))
        output.info = {"label": "qiskit"}
        return output


class DampingAlpha(cirq.ops.gate_features.SingleQubitGate):
    """A cirq gate class for the first damping Kraus operator."""

    def __init__(self, p):
        self.p = p

    def _unitary_(self):
        return np.diag([1, np.sqrt(1 - self.p)])

    def __str__(self):
        return f"Da({self.p})"


class DampingBeta(cirq.ops.gate_features.SingleQubitGate):
    """A cirq gate class for the first damping Kraus operator."""

    def __init__(self, p):
        self.p = p

    def _unitary_(self):
        matrix = np.zeros((2, 2))
        matrix[0, 1] = np.sqrt(self.p)
        return matrix

    def __str__(self):
        return f"Db({self.p})"


class MCTGate(object):
    """
    An Orquestra gate class that deals with synthesis of a Multi Toffoli Gate
    Although this inherits the Gate class it has extra attributes fundamentally related to
    the fact that the Multi Toffoli Gate is a circuit rather a gate applied natively on a device

    Attributes:
    all_circuit_qubits (List): Needed to apply the mct gate from qiskit. In general we need more qubits than those
                                 assigned as control and target qubits. We thus get all qubits in the circuit in order
                                 to use them as possible ancilla qubits.

    control_qubits (list): This list contains integer labels for the qubits thats will act as the
    control qubits. e.g [1, 2, 3, 4]
    target_qubits (list): This is a list that contains integer labels for the qubits that are target qubits

    all_qubits (list): A list of zmachine.core.qubit.Qubit This is necessary to determine what qubits
    are the ancilla

    ancilla_qubits (list): A list that contians integer labels for all the qubits labels that are the ancilla. These are needed since the MCT synthesis
    requires in general addition of ancilla qubits

    ccx_list (list): A list that contains CCX orquestra gates that represent the decompostion of the MCT gate

    ccx_decomposition (qiskit circuit): Contains the Toffoli Decomposition which is the output from the mct gate

    qiskit_ctrl_q (list): List of qiskit qubits that are control qubits for the MCT gate
    qiskit_targ_q (list): List of qiskit qubits that are targ qubits for the MCT gate

    qiskit_ancilla_q (list): List of qiskit qubits that are ancilla qubits for the mct gate synthesis


    """

    def __init__(self, all_circuit_qubits, control_qubits=None, target_qubits=None):
        if isinstance(control_qubits, list) and isinstance(target_qubits, list):
            if control_qubits is None and target_qubits is None:
                self.control_qubits = []
                self.target_qubits = []
            self.control_qubits = control_qubits
            self.target_qubits = target_qubits
            self.all_circuit_qubits = all_circuit_qubits
            self.ancilla_qubits = []
            self.ccx_list = []
            self.qiskit_data = []
            for i in range(len(self.all_circuit_qubits)):
                if i not in self.control_qubits and i not in self.target_qubits:
                    self.ancilla_qubits.append(i)
        else:
            raise TypeError(
                "Expecting a list integer labels for control and target qubits"
            )
        # This is the ibm circuit object with all the toffoli gates
        self.ccx_decomposition = qiskit.QuantumCircuit(
            len(self.all_circuit_qubits), len(self.all_circuit_qubits)
        )
        self.qiskit_ctrl_q = None
        self.qiskit_targ_q = None
        self.qiskit_ancilla_q = None
        self._create_gate_list()

    def _synthesize_circuit(self):
        """
        Produces the gate decomposition using qiskit
        """
        self.ccx_decomposition.mct(
            self.qiskit_ctrl_q, self.qiskit_targ_q, self.qiskit_ancilla_q
        )

    def _create_qiskit_qubits(self):
        """
        Producing list of qiskit qubit objects
        """
        qreg = QuantumRegister(len(self.all_circuit_qubits), "q")
        self.qiskit_ctrl_q = [qreg[i] for i in self.control_qubits]
        self.qiskit_targ_q = qreg[self.target_qubits[0]]
        self.qiskit_ancilla_q = [qreg[i] for i in self.ancilla_qubits]

    def _create_gate_list(self):
        # First create qiskit objects
        self._create_qiskit_qubits()

        # Synthesize circuit
        self._synthesize_circuit()
        for i in range(len(self.ccx_decomposition.data)):
            self.qiskit_data += list(self.ccx_decomposition[i])


class PhaseOracle(MCTGate):
    """
    This applies the Multicontrol phase gate
    Attributes:
    phase(numpy float): The phase angle for oracle to be implemented
    """

    def __init__(
        self, phase, all_circuit_qubits, control_qubits=None, target_qubits=None
    ):
        self.phase = phase
        if control_qubits is None and target_qubits is None:
            self.control_qubits = []
            self.target_qubits = []
        super().__init__(
            all_circuit_qubits,
            control_qubits=control_qubits,
            target_qubits=target_qubits,
        )

    def _synthesize_circuit(self):
        """
        Produces the circuit that represents the multi control phase gate
        """
        self.ccx_decomposition.mcu1(self.phase, self.qiskit_ctrl_q, self.qiskit_targ_q)


class MCRY(MCTGate):
    """
    This applies the MulltiControl Y rotation
    Attributes:
    phase(numpy float): The phase angle for oracle to be implemente
    """

    def __init__(
        self, phase, all_circuit_qubits, control_qubits=None, target_qubits=None
    ):
        self.phase = phase
        if control_qubits is None and target_qubits is None:
            self.control_qubits = []
            self.target_qubits = []
        super().__init__(
            all_circuit_qubits,
            control_qubits=control_qubits,
            target_qubits=target_qubits,
        )

    def _synthesize_circuit(self):
        """
        Produces the circuit that represents the multicontrol y rotation
        """
        self.ccx_decomposition.mcry(
            self.phase, self.qiskit_ctrl_q, self.qiskit_targ_q, self.qiskit_ancilla_q
        )
