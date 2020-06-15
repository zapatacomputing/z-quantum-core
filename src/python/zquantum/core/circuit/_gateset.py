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
