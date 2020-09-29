import copy


def compile(gates):
    new_gates = reduce_gates(gates)
    while len(new_gates) < len(gates):
        gates = new_gates
        new_gates = reduce_gates(gates)
    # print(
    #     "Compiled gates: ",
    #     [(gate.name, [qubit.index for qubit in gate.qubits]) for gate in gates],
    # )
    return gates


def reduce_gates(gates):
    new_gates = []
    current_index = 0

    while current_index < len(gates):
        comparison_index = current_index + 1
        current_gate_was_reduced = False
        # print("\nAttempting to reduce: ", gates[current_index])

        while comparison_index < len(gates):

            # print("with: ", gates[comparison_index])
            can_reduce, reduced_gates = check_if_gates_reduce(
                gates[current_index], gates[comparison_index]
            )

            if can_reduce:
                # print("successfully reduced gate")
                new_gates += reduced_gates
                gates.remove(gates[comparison_index])
                current_gate_was_reduced = True
                break
            elif check_if_gates_commute(gates[current_index], gates[comparison_index]):
                # print("gate commuted")
                comparison_index += 1
            else:
                # print("gates did not commute")
                # print("could not reduce current gate")
                break

        if not current_gate_was_reduced:
            # print("adding current gate to new gates")
            new_gates.append(gates[current_index])
        current_index += 1

    # print([gate.name for gate in new_gates])
    return new_gates


def check_if_gates_reduce(gate1, gate2):
    # print("Attempting to reduce: {}, {}".format(gate1, gate2))
    if gate1.name == gate2.name and gate1.qubits == gate2.qubits:
        if gate1.name in ["X", "Y", "Z", "H", "CNOT", "SWAP"]:
            return True, []
        elif gate1.name in ["Rx", "Ry", "Rz"]:
            new_gate = copy.deepcopy(gate1)
            new_gate.params += gate2.params
            return True, [new_gate]
    return False, None


def check_if_gates_commute(gate1, gate2):
    return False


#         for current_index, current_gate in enumerate(gates):
#             for comparison_index, comparison_gate in enumerate(
#                 gates[current_index + 1 : :]
#             ):
#                 can_reduce, replacement_gates = check_if_gates_reduce(
#                     current_gate, comparison_gate
#                 )

#                 if can_reduce:
#                     successfully_removed_gates = True
#                     new_gates += replacement_gates


# [X, X, Rx, Ry, Y, Z, CNOT, X]
