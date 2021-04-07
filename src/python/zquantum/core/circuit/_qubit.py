"""Base class for qubit"""

import cirq
import pyquil


class Qubit:
    """Class for storing information associated with a qubit.

    Attributes:
        index: int, (int, int)
            The index or set of indices labelling a qubit.
        label: string
            A string which specifies the native package that the Qubit is generated from.
        info: dictionary
            Additional information depending on the 'label' of the Qubit. See the Qubit.from_...
            methods for details.
    """

    def __init__(self, index=-1):

        self.index = index

        # optional attributes
        self.info = {"label": "none"}

    def __str__(self):
        return f"qubit {self.index}"

    def __repr__(self):
        return f"zquantum.core.circuit.Qubit(index={self.index})"

    def to_dict(self):
        return {"index": self.index, "info": self.info}

    @classmethod
    def from_dict(cls, dict):
        qubit = cls(dict["index"])
        qubit.info = dict["info"]
        return qubit

    @classmethod
    def from_pyquil(cls, pyquil_qubit):
        """Converts a pyquil Qubit object to core.qubit.Qubit object.

        Args:
            pyquil_qubit: pyquil.quilbase.Qubit
                The qubit object in pyquil.
            index: int
                The index of the Qubit object to be generated.

        Returns:
            A core.qubit.Qubit object. Entries include
                label: string
                    Name of the package that generated the qubit object. Here it is equal to 'pyquil'
        """

        output = cls()
        output.info["label"] = "pyquil"
        if isinstance(pyquil_qubit, pyquil.quilatom.Qubit):
            output.index = pyquil_qubit.index
        else:
            raise TypeError(f"Input item {pyquil_qubit} not a pyquil Qubit object")
        return output

    @classmethod
    def from_cirq(cls, cirq_qubit, index):
        """Converts a qubit object in cirq to core.qubit.Qubit object.

        Args:
            cirq_qubit: cirq.LineQubit, cirq.GridQubit
                Input cirq qubit object.
            index: int
                The integer index of the qubit

        Returns:
            A core.qubit.Qubit object. Here the attributes are stored as
                label: string
                    The name of the package that generates the qubit object. In this case
                    it is 'cirq'.
                index: int
                    The index of the Qubit is an integer, while the full key is stored
                    in the 'info' entry of the Qubit object.
                info: dictionary
                    The key 'QubitType' holds the value of which kind of qubit cirq_qubit is.
                    The key 'QubitKey' holds the full key (row, col) if cirq_qubit is a
                    cirq.GridQubit object.
        """

        output = cls(index)
        output.info["label"] = "cirq"
        if isinstance(cirq_qubit, cirq.GridQubit):
            qubit_key = lambda q: (q.row, q.col)
            output.info["QubitType"] = "GridQubit"
            output.info["QubitKey"] = qubit_key(cirq_qubit)
        elif isinstance(cirq_qubit, cirq.LineQubit):
            qubit_key = lambda q: q.x
            output.info["QubitType"] = "LineQubit"
            output.info["QubitKey"] = qubit_key(cirq_qubit)
        else:
            raise ValueError("Qubit type {} not yet supported".format(type(cirq_qubit)))

        return output

    @classmethod
    def from_qiskit(cls, qiskit_qubit, index):
        """Converts a qubit object in qiskit to core.qubit.Qubit object.

        Args:
            qiskit_qubit: (qiskit.Qubit, int)
                A tuple representing the qubit
            index: int
                Index to be assigned to the output core.qubit.Qubit object

        Returns:
            A core.qubit.Qubit object. Here the attributes are stored as
                index: int
                    Here each qubit is assigned a unique index which is an integer.
                info: dictionary
                    label: string
                        Name of the package that generates the qubit object. Here it
                        is 'qiskit'
                    qreg: string
                        Holds the string representing register corresponding to the qubit.
                    num: int
                        Holds the index of the qubit inside the quantum register.
        """

        output = cls(index)
        output.info["label"] = "qiskit"
        output.info["num"] = index

        return output
