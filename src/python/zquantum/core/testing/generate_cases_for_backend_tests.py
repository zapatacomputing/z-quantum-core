from operator import mul

import sympy
from sympy.physics.quantum import TensorProduct

# This script is used to generate test cases for tests in `core/interfaces/backend_test.py`
# In order to use just run this python file with python3 generate_cases_for_backend_tests.py
# and then copy-paste the console output to the appropriate place in `test_cases_for_backend_tests.py`

# If someone would like to add new testcases to core/interfaces/backend_test.py,
# e.g. testing new gate or different initial state, they could use this script to do that.
# (or they can do that manually).

I = sympy.Matrix([[1, 0], [0, 1]])
H = sympy.Matrix(
    [[1 / sympy.sqrt(2), 1 / sympy.sqrt(2)], [1 / sympy.sqrt(2), -1 / sympy.sqrt(2)]]
)
X = sympy.Matrix([[0, 1], [1, 0]])
Y = sympy.Matrix([[0, -sympy.I], [sympy.I, 0]])
Z = sympy.Matrix([[1, 0], [0, -1]])
S = sympy.Matrix([[1, 0], [0, sympy.I]])
T = sympy.Matrix([[1, 0], [0, (1 + sympy.I) / sympy.sqrt(2)]])

CNOT = sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
SWAP = sympy.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
ISWAP = sympy.Matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])

II = TensorProduct(I, I)
IH = TensorProduct(H, I)
HI = TensorProduct(I, H)

HH = TensorProduct(H, H)
XX = TensorProduct(X, X)
YY = TensorProduct(Y, Y)
ZZ = TensorProduct(Z, Z)
IX = TensorProduct(X, I)
ZI = TensorProduct(I, Z)

single_qubit_initial_states = [(I, "I"), (H, "H")]
single_qubit_operators = [I, X, Y, Z]

two_qubit_initial_states = [
    (II, ["I", "I"]),
    (IH, ["I", "H"]),
    (HI, ["H", "I"]),
    (HH, ["H", "H"]),
]
two_qubit_operators = [
    (II, "[]"),
    (XX, "[X0 X1]"),
    (YY, "[Y0 Y1]"),
    (ZZ, "[Z0 Z1]"),
    (IX, "[X1]"),
    (ZI, "[Z0]"),
]


def generate_cases_1_qubit_wavefunction(matrix, matrix_name, angles):
    for initial_matrix, initial_matrix_name in single_qubit_initial_states:
        for angle in angles:
            circuit = mul(matrix, initial_matrix)
            new_circuit = circuit.subs("theta", angle)
            amplitudes = new_circuit * sympy.Matrix([[1], [0]])
            gate_names_string = f'["{initial_matrix_name}", "{matrix_name}", '
            angle_string = f"[{angle}], ".replace("pi", "np.pi")
            amplitude_string = (
                str([amplitudes[0], amplitudes[1]])
                .replace("sqrt", "np.sqrt")
                .replace("pi", "np.pi")
                .replace("1.0*I", "1.0j")
                .replace("*I", "*1.0j")
                .replace("exp", "np.exp")
            )
            print(gate_names_string + angle_string + amplitude_string + "],")


def generate_cases_1_qubit_exp_vals(matrix, matrix_name):
    for initial_matrix, initial_matrix_name in single_qubit_initial_states:
        outputs = []
        for operator in single_qubit_operators:
            circuit_ket = mul(matrix, initial_matrix) * sympy.Matrix([[1], [0]])
            circuit_bra = sympy.conjugate(sympy.Transpose(circuit_ket))
            expectation_value = mul(circuit_bra, mul(operator, circuit_ket))[0]
            outputs.append(sympy.simplify(expectation_value))

        gate_names_string = f'["{initial_matrix_name}", "{matrix_name}", '
        exp_vals_string = (
            "[{}, {}, {}, {}]".format(*outputs)
            .replace("sqrt", "np.sqrt")
            .replace("pi", "np.pi")
            .replace("1.0*I", "1.0j")
            .replace("*I", "*1.0j")
        )
        print(gate_names_string + exp_vals_string + "],")


def generate_cases_1_qubit_exp_vals_with_angles(matrix, matrix_name, angles):
    for initial_matrix, initial_matrix_name in single_qubit_initial_states:
        for angle in angles:
            outputs = []
            for operator in single_qubit_operators:
                circuit_ket = mul(matrix, initial_matrix) * sympy.Matrix([[1], [0]])
                circuit_ket = circuit_ket.subs("theta", angle)
                circuit_bra = sympy.conjugate(sympy.Transpose(circuit_ket))
                expectation_value = mul(circuit_bra, mul(operator, circuit_ket))[0]
                outputs.append(sympy.simplify(expectation_value))

            gate_names_string = f'["{initial_matrix_name}", "{matrix_name}", '
            angle_string = f"[{angle}], ".replace("pi", "np.pi")
            exp_vals_string = (
                "[{}, {}, {}, {}]".format(*outputs)
                .replace("sqrt", "np.sqrt")
                .replace("pi", "np.pi")
                .replace("1.0*I", "1.0j")
                .replace("*I", "*1.0j")
            )
            print(gate_names_string + angle_string + exp_vals_string + "],")


def generate_cases_2_qubits_wavefunction(matrix, matrix_name, angles):
    for initial_matrix, initial_matrix_names in two_qubit_initial_states:
        for angle in angles:
            circuit = mul(matrix, initial_matrix)
            new_circuit = circuit.subs("theta", angle)
            amplitudes = new_circuit * sympy.Matrix([[1], [0], [0], [0]])
            gate_names_string = '[["{}", "{}"], "{}", '.format(
                initial_matrix_names[0], initial_matrix_names[1], matrix_name
            )
            angle_string = f"[{angle}], ".replace("pi", "np.pi")
            amplitude_string = (
                str([amplitudes[0], amplitudes[1], amplitudes[2], amplitudes[3]])
                .replace("sqrt", "np.sqrt")
                .replace("pi", "np.pi")
                .replace("1.0*I", "1.0j")
                .replace("*I", "*1.0j")
            )
            print(gate_names_string + angle_string + amplitude_string + "],")


def generate_cases_2_qubits_exp_vals(matrix, matrix_name):
    for initial_matrix, initial_matrix_names in two_qubit_initial_states:
        outputs = []
        operator_names = []
        for operator, operator_name in two_qubit_operators:
            operator_names.append(f'"{operator_name}"')
            circuit_ket = mul(matrix, initial_matrix) * sympy.Matrix(
                [[1], [0], [0], [0]]
            )
            circuit_bra = sympy.conjugate(sympy.Transpose(circuit_ket))
            expectation_value = mul(circuit_bra, mul(operator, circuit_ket))[0]

            outputs.append(sympy.simplify(expectation_value))
        operator_names_string = "[" + ", ".join(operator_names) + "], "
        gate_names_string = '[["{}", "{}"], "{}", '.format(
            initial_matrix_names[0], initial_matrix_names[1], matrix_name
        )
        exp_vals_string = "["
        for output in outputs:
            exp_vals_string += (
                f"{output},".replace("sqrt", "np.sqrt")
                .replace("pi", "np.pi")
                .replace("1.0*I", "1.0j")
                .replace("*I", "*1.0j")
            )
        exp_vals_string += "]"

        print(gate_names_string + operator_names_string + exp_vals_string + "],")


def generate_cases_2_qubits_exp_vals_with_angles(matrix, matrix_name, angles):
    for initial_matrix, initial_matrix_names in two_qubit_initial_states:
        for angle in angles:
            outputs = []
            operator_names = []
            for operator, operator_name in two_qubit_operators:
                operator_names.append(f'"{operator_name}"')
                circuit_ket = mul(matrix, initial_matrix) * sympy.Matrix(
                    [[1], [0], [0], [0]]
                )
                circuit_ket = circuit_ket.subs("theta", angle)
                circuit_bra = sympy.conjugate(sympy.Transpose(circuit_ket))
                expectation_value = mul(circuit_bra, mul(operator, circuit_ket))[0]

                outputs.append(sympy.simplify(expectation_value))
            operator_names_string = "[" + ", ".join(operator_names) + "], "
            gate_names_string = '[["{}", "{}"], "{}", '.format(
                initial_matrix_names[0], initial_matrix_names[1], matrix_name
            )
            angle_string = f"[{angle}], ".replace("pi", "np.pi")
            exp_vals_string = "["
            for output in outputs:
                exp_vals_string += (
                    f"{output},".replace("sqrt", "np.sqrt")
                    .replace("pi", "np.pi")
                    .replace("1.0*I", "1.0j")
                    .replace("*I", "*1.0j")
                )
            exp_vals_string += "]"

            print(
                gate_names_string
                + angle_string
                + operator_names_string
                + exp_vals_string
                + "],"
            )


def main():
    theta = sympy.Symbol("theta")
    Rx = sympy.Matrix(
        [
            [sympy.cos(theta / 2), -1j * sympy.sin(theta / 2)],
            [-1j * sympy.sin(theta / 2), sympy.cos(theta / 2)],
        ]
    )
    Ry = sympy.Matrix(
        [
            [sympy.cos(theta / 2), -sympy.sin(theta / 2)],
            [sympy.sin(theta / 2), sympy.cos(theta / 2)],
        ]
    )
    Rz = sympy.Matrix(
        [
            [sympy.cos(theta / 2) - 1j * sympy.sin(theta / 2), 0],
            [0, sympy.cos(theta / 2) + 1j * sympy.sin(theta / 2)],
        ]
    )
    PHASE = sympy.Matrix([[1, 0], [0, sympy.cos(theta) + 1j * sympy.sin(theta)]])
    RH_phase_factor = sympy.exp(1j * theta / 2)
    RH = sympy.Matrix(
        [
            [
                RH_phase_factor
                * (sympy.cos(theta / 2) - 1j / sympy.sqrt(2) * sympy.sin(theta / 2)),
                RH_phase_factor * -1j / sympy.sqrt(2) * sympy.sin(theta / 2),
            ],
            [
                RH_phase_factor * -1j / sympy.sqrt(2) * sympy.sin(theta / 2),
                RH_phase_factor
                * (sympy.cos(theta / 2) + 1j / sympy.sqrt(2) * sympy.sin(theta / 2)),
            ],
        ]
    )

    CPHASE = sympy.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, sympy.cos(theta) + 1j * sympy.sin(theta)],
        ]
    )
    XX = sympy.Matrix(
        [
            [sympy.cos(theta), 0, 0, -1j * sympy.sin(theta)],
            [0, sympy.cos(theta), -1j * sympy.sin(theta), 0],
            [0, -1j * sympy.sin(theta), sympy.cos(theta), 0],
            [-1j * sympy.sin(theta), 0, 0, sympy.cos(theta)],
        ]
    )
    YY = sympy.Matrix(
        [
            [sympy.cos(theta), 0, 0, 1j * sympy.sin(theta)],
            [0, sympy.cos(theta), -1j * sympy.sin(theta), 0],
            [0, -1j * sympy.sin(theta), sympy.cos(theta), 0],
            [1j * sympy.sin(theta), 0, 0, sympy.cos(theta)],
        ]
    )
    ZZ = sympy.Matrix(
        [
            [sympy.cos(theta) - 1j * sympy.sin(theta), 0, 0, 0],
            [0, sympy.cos(theta) + 1j * sympy.sin(theta), 0, 0],
            [0, 0, sympy.cos(theta) + 1j * sympy.sin(theta), 0],
            [0, 0, 0, sympy.cos(theta) - 1j * sympy.sin(theta)],
        ]
    )

    XY = XX * YY
    XY.simplify()
    angles = [-sympy.pi / 2, 0, sympy.pi / 5, sympy.pi / 2, sympy.pi]
    print("**" * 10)
    print("WAVEFUNCTION")
    print("-" * 10)
    print("1 qubit gates")
    print("-" * 10)
    generate_cases_1_qubit_wavefunction(Rx, "Rx", angles)
    generate_cases_1_qubit_wavefunction(Ry, "Ry", angles)
    generate_cases_1_qubit_wavefunction(Rz, "Rz", angles)
    generate_cases_1_qubit_wavefunction(PHASE, "PHASE", angles)
    generate_cases_1_qubit_wavefunction(RH, "RH", angles)
    print("-" * 10)
    print("2 qubit gates")
    print("-" * 10)
    generate_cases_2_qubits_wavefunction(CPHASE, "CPHASE", angles)
    generate_cases_2_qubits_wavefunction(XX, "XX", angles)
    generate_cases_2_qubits_wavefunction(YY, "YY", angles)
    generate_cases_2_qubits_wavefunction(ZZ, "ZZ", angles)
    generate_cases_2_qubits_wavefunction(XY, "XY", angles)

    print("**" * 10)
    print("EXP VALS WITHOUT ANGLES")
    print("-" * 10)
    print("1 qubit gates")
    print("-" * 10)
    generate_cases_1_qubit_exp_vals(X, "X")
    generate_cases_1_qubit_exp_vals(Y, "Y")
    generate_cases_1_qubit_exp_vals(Z, "Z")
    generate_cases_1_qubit_exp_vals(H, "H")
    generate_cases_1_qubit_exp_vals(S, "S")
    generate_cases_1_qubit_exp_vals(T, "T")
    print("-" * 10)
    print("2 qubit gates")
    print("-" * 10)
    generate_cases_2_qubits_exp_vals(CNOT, "CNOT")
    generate_cases_2_qubits_exp_vals(SWAP, "SWAP")
    generate_cases_2_qubits_exp_vals(ISWAP, "ISWAP")
    generate_cases_2_qubits_exp_vals(CZ, "CZ")

    print("**" * 10)
    print("EXP VALS WITH ANGLES")
    print("-" * 10)
    print("1 qubit gates")
    print("-" * 10)
    generate_cases_1_qubit_exp_vals_with_angles(Rx, "Rx", angles)
    generate_cases_1_qubit_exp_vals_with_angles(Ry, "Ry", angles)
    generate_cases_1_qubit_exp_vals_with_angles(Rz, "Rz", angles)
    generate_cases_1_qubit_exp_vals_with_angles(PHASE, "PHASE", angles)
    generate_cases_1_qubit_exp_vals_with_angles(RH, "RH", angles)
    print("-" * 10)
    print("2 qubit gates")
    print("-" * 10)
    generate_cases_2_qubits_exp_vals_with_angles(CPHASE, "CPHASE", angles)
    generate_cases_2_qubits_exp_vals_with_angles(XX, "XX", angles)
    generate_cases_2_qubits_exp_vals_with_angles(YY, "YY", angles)
    generate_cases_2_qubits_exp_vals_with_angles(ZZ, "ZZ", angles)
    generate_cases_2_qubits_exp_vals_with_angles(XY, "XY", angles)


if __name__ == "__main__":
    main()
