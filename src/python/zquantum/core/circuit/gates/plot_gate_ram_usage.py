import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from zquantum.core.circuit.gate import Gate
import sympy
import copy
import sys


def get_memory_for_two_qubits(number_of_gates):
    qubits = (
        0,
        1,
    )
    matrix = sympy.Matrix(
        [
            [
                sympy.cos(sympy.Symbol("theta") / 2),
                -1 * sympy.sin(sympy.Symbol("theta") / 2),
                sympy.cos(sympy.Symbol("theta") / 2),
                -1 * sympy.sin(sympy.Symbol("theta") / 2),
            ],
            [
                -1 * sympy.sin(sympy.Symbol("theta") / 2),
                sympy.cos(sympy.Symbol("theta") / 2),
                sympy.cos(sympy.Symbol("theta") / 2),
                -1 * sympy.sin(sympy.Symbol("theta") / 2),
            ],
            [
                sympy.cos(sympy.Symbol("theta") / 2),
                -1 * sympy.sin(sympy.Symbol("theta") / 2),
                sympy.cos(sympy.Symbol("theta") / 2),
                -1 * sympy.sin(sympy.Symbol("theta") / 2),
            ],
            [
                -1 * sympy.sin(sympy.Symbol("theta") / 2),
                sympy.cos(sympy.Symbol("theta") / 2),
                sympy.cos(sympy.Symbol("theta") / 2),
                -1 * sympy.sin(sympy.Symbol("theta") / 2),
            ],
        ]
    )
    gate = Gate(matrix, qubits)

    gate_list = [copy.deepcopy(gate) for _ in range(int(number_of_gates))]
    memoryUse = sys.getsizeof(gate_list)
    return memoryUse


def get_memory_for_single_qubit(number_of_gates):
    qubits = (0,)
    matrix = sympy.Matrix(
        [
            [
                sympy.cos(sympy.Symbol("theta") / 2),
                -1 * sympy.sin(sympy.Symbol("theta") / 2),
            ],
            [
                -1 * sympy.sin(sympy.Symbol("theta") / 2),
                sympy.cos(sympy.Symbol("theta") / 2),
            ],
        ]
    )
    gate = Gate(matrix, qubits)

    gate_list = [copy.deepcopy(gate) for _ in range(int(number_of_gates))]
    memoryUse = sys.getsizeof(gate_list)
    return memoryUse


def get_data_usage(number_of_gates, number_of_trials, two_qubits=False):
    if two_qubits:
        function = get_memory_for_two_qubits
    else:
        function = get_memory_for_single_qubit

    memory_usage = [function(number_of_gates) for _ in range(number_of_trials)]
    average_memory_use = np.average([memory_usage])
    return average_memory_use


fig = plt.figure()
ax = plt.axes()

MAX_NUMBER_OF_GATES = 10000

numbers_of_gates = np.linspace(0, MAX_NUMBER_OF_GATES, num=10)

amount_of_memory_for_single_qubit_gates = [
    get_data_usage(number_of_gates, 1) for number_of_gates in numbers_of_gates
]

amount_of_memory_for_two_qubit_gates = [
    get_data_usage(number_of_gates, 1, two_qubits=True)
    for number_of_gates in numbers_of_gates
]

print(numbers_of_gates)
print(amount_of_memory_for_single_qubit_gates)
print(amount_of_memory_for_two_qubit_gates)

ax.plot(
    numbers_of_gates,
    amount_of_memory_for_single_qubit_gates,
    color="orange",
    label="Memory Usage for Single Qubit Gates",
)
ax.plot(
    numbers_of_gates,
    amount_of_memory_for_two_qubit_gates,
    color="blue",
    label="Memory Usage for Two Qubit Gates",
)

legend = ax.legend(loc="lower right")

ax.set_ylabel("Amount of RAM (Mb)")
ax.set_xlabel("Number of Gates")
ax.set_xlim(
    1, MAX_NUMBER_OF_GATES,
)
plt.show()

