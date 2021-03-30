"""Public API for the ZQuantum circuits.

Allows:

- defining quantum circuits with gates applied to qubits
- set of built-in gates (see imports in this `__init__`)
- import/export from/to Cirq, PyQuil & Qiskit circuits
- circuit (de)serialization to/from JSON-compatible dicts

Examples:

    Defining a circuit with a NOT gate on qubit 0 & Hadamard gate on qubit 1::
        circuit = Circuit()
        circuit += X(0)
        circuit += H(1)

    Adding 2-qubit gates::
        circuit += CNOT(0, 4)

    Adding parametrized gates::
        circuit += RX(np.pi / 2)(0)
        circuit += RX(sympy.sympify("theta / 2"))(1)

    Adding 2-qubit parametrized gates::
        circuit += CPHASE(0.314)(3, 2)

    Adding a built-in gate by its name::
        circuit += builtin_gate_by_name("X")(0)
        circuit += builtin_gate_by_name("RX")(np.pi * 1.5)(1)

    Binding parameters::
        circuit = circuit.bind({sympy.Symbol("theta"): -np.pi / 5})

    Iterating over circuit contents::
        for gate_op in circuit.operations:
            name = gate_op.gate.name
            params = gate_op.gate.params
            qubits = gate_op.qubit_indices
            print(f"{name} with params {params} applied to {qubits}")

    Making a different circuit (e.g. shifting gates by 1 qubit):: 
        new_circuit = Circuit(
            operations=[
                gate_op.gate(*[qubit + 1 for qubit in gate_op.qubits])
                gate_op for gate_op in circuit.operations
            ],
            n_qubits=circuit.n_qubits
        )

    Conversion to other frameworks::
        export_to_cirq(circuit)
        circuit2 = import_from_cirq(cirq_circuit)

        export_to_pyquil(circuit)
        circuit3 = import_from_pyquil(pyquil_program)

        export_to_qiskit(circuit)
        circuit4 = import_from_qiskit(qiskit_quantum_circuit)


See imports in this module for a list of built-in gates.
"""

from ._gates import (
    Gate,
    GateOperation,
    MatrixFactoryGate,
    ControlledGate,
    Dagger,
    CustomGateDefinition,
)

from ._circuit import Circuit

from ._builtin_gates import (
    X,
    Y,
    Z,
    H,
    I,
    S,
    T,
    RX,
    RY,
    RZ,
    PHASE,
    CNOT,
    CZ,
    SWAP,
    ISWAP,
    CPHASE,
    XX,
    YY,
    ZZ,
    XY,
    builtin_gate_by_name
)

from ._serde import (
    to_dict,
    circuit_from_dict,
)

from .conversions.cirq_conversions import (
    export_to_cirq,
    import_circuit_from_cirq,
)

from .conversions.pyquil_conversions import (
    export_to_pyquil,
    import_from_pyquil,
)

from .conversions.qiskit_conversions import (
    export_to_qiskit,
    import_from_qiskit,
)
