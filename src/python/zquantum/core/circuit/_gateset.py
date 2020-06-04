"""Standard gate set for Orquestra.

Convention for Gate objects currently supported:

Gate.name: string
    Name of the quantum gate.
Gate.qubits: [Qubit]
    List of Qubit objects, whose length depends on the number of qubits that the 
    gate acts on.
Gate.params: [float]
    List of gate parameters. 
    For discrete gates this entry is empty. 
    For continuous gates this entry contains parameter(s) needed to specify the gate.
        For x, y, z rotation gates and their controlled variants this entry contains 
        the angle parameter.
"""

# All gates supported in Orquestra
COMMON_GATES = [
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
    "CNOT",
    "CZ",
    "CPHASE",
    "SWAP",
]
UNIQUE_GATES = [
    "ZXZ",
    "RH",
    "XX",
    "YY",
    "ZZ",
    "U1ex",
    "Da",
    "Db",
    "MEASURE",
    "BARRIER",
]  # gates unique to specific packages
ALL_GATES = COMMON_GATES + UNIQUE_GATES


# All gates natively supported in pyquil
# (generated from pyquil.gates.__dict__['__all__'])
PYQUIL_GATES = [
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "T",
    "PHASE",
    "RX",
    "RY",
    "RZ",
    "CZ",
    "CNOT",
    "CCNOT",
    "CPHASE00",
    "CPHASE01",
    "CPHASE10",
    "CPHASE",
    "SWAP",
    "CSWAP",
    "ISWAP",
    "PSWAP",
    "WAIT",
    "RESET",
    "NOP",
    "HALT",
    "MEASURE",
    "TRUE",
    "FALSE",
    "NOT",
    "AND",
    "OR",
    "MOVE",
    "EXCHANGE",
    "IOR",
    "XOR",
    "NEG",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "EQ",
    "GT",
    "GE",
    "LE",
    "LT",
    "LOAD",
    "STORE",
    "CONVERT",
    "Gate",
]

# All gates natively supported in cirq
CIRQ_GATES = [
    "PhasedXPowGate",
    "CNOT",
    "CNotPowGate",
    "CZ",
    "CZPowGate",
    "H",
    "HPowGate",
    "ISWAP",
    "ISwapPowGate",
    "measure",
    "measure_each",
    "MeasurementGate",
    "XPowGate",
    "YPowGate",
    "ZPowGate",
    "Rx",
    "Ry",
    "Rz",
    "S",
    "SWAP",
    "SwapPowGate",
    "T",
    "X",
    "Y",
    "Z",
    "XX",
    "XXPowGate",
    "YY",
    "YYPowGate",
    "ZZ",
    "ZZPowGate",
    "PhasedXPowGate",
    "CCX",
    "CCXPowGate",
    "CCZ",
    "CCZPowGate",
    "CSWAP",
    "CSwapGate",
    "FREDKIN",
    "TOFFOLI",
]

# All gates natively support in qiskit
QISKIT_GATES = [
    "ubase",
    "u2",
    "h",
    "cxbase",
    "cx",
    "u1",
    "t",
    "ccx",
    "cswap",
    "cx_base",
    "s",
    "cy",
    "cz",
    "swap",
    "iden",
    "sdg",
    "tdg",
    "u0",
    "u3",
    "u_base",
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "cu1",
    "ch",
    "crz",
    "cu3",
    "rzz",
    "measure",
    "barrier",
]
