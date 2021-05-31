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
                for gate_op in circuit.operations
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

    (De)serialization::
        to_dict(circuit)
        circuit5 = circuit_from_dict(dict5)


Defining new gates
------------------

To use a gate that isn't already covered by built-in ones you can define a custom gate
or extend the set of the built-in ones and file a PR to z-quantum-core.

Using custom gates::
    custom_a = circuits.CustomGateDefinition(
        gate_name="custom_a",  # names need to be unique
        matrix=sympy.Matrix(
            [
                [-1, 0],
                [0, 1],
            ]
        ),
        params_ordering=(),
    )

    custom_b = circuits.CustomGateDefinition(
        gate_name="custom_b",
        matrix=sympy.Matrix(
            [
                [0, sympy.Symbol("theta") * 2],
                [sympy.Symbol("gamma") + 3, 1],
            ]
        ),
        params_ordering=(sympy.Symbol("gamma"), sympy.Symbol("theta")),
    )

    circuit = Circuit()
    circuit += custom_a()(0)
    circuit += custom_b(np.pi, np.pi / 2)(0)


Extending built-in gates requires:

- Adding its definition to `zquantum.core.circuits._builtin_gates`. Refer to other
    1- or multi-qubit, parametric/nonparametric gates there to see how it's been done
    for other gates.

- Adding its matrix to `zquantum.core.circuits._matrices`.

- Adding tests for conversion to other frameworks in:
    - `zquantum.core.conversions.cirq_conversions_test`
    - `zquantum.core.conversions.pyquil_conversions_test`
    - `zquantum.core.conversions.qiskit_conversions_test`

- Implement conversions. Some might work out of the box, e.g. if there's a gate with the
    same name defined in PyQuil our converters will use it by default without need for
    explicit mappings.
"""

from ._builtin_gates import (
    CNOT,
    CPHASE,
    CZ,
    ISWAP,
    PHASE,
    RH,
    RX,
    RY,
    RZ,
    SWAP,
    U3,
    XX,
    XY,
    YY,
    ZZ,
    GatePrototype,
    H,
    I,
    S,
    T,
    X,
    Y,
    Z,
    builtin_gate_by_name,
)
from ._circuit import Circuit
from ._compatibility import new_circuit_from_old_circuit
from ._gates import (
    ControlledGate,
    CustomGateDefinition,
    Dagger,
    Gate,
    GateOperation,
    MatrixFactoryGate,
)
from ._generators import add_ancilla_register, create_layer_of_gates
from ._serde import (
    circuit_from_dict,
    circuitset_from_dict,
    load_circuit,
    load_circuitset,
    save_circuit,
    save_circuitset,
    to_dict,
)
from ._testing import create_random_circuit
from ._wavefunction_operations import MultiPhaseOperation
from .conversions.cirq_conversions import export_to_cirq, import_from_cirq
from .conversions.pyquil_conversions import export_to_pyquil, import_from_pyquil
from .conversions.qiskit_conversions import export_to_qiskit, import_from_qiskit
