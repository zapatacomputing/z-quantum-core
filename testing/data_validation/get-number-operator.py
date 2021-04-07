import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 1

for key in workflowresult.keys():
    assert workflowresult[key]["class"] == "get-number-operator"
    assert workflowresult[key]["inputParam:n-qubits"] == "4"
    assert workflowresult[key]["inputParam:n-particles"] == "None"

    assert workflowresult[key]["number-op"]["schema"] == "zapata-v1-interaction_op"

    assert workflowresult[key]["number-op"]["constant"]["real"] == 0

    for array in workflowresult[key]["number-op"]["one_body_tensor"]["imag"]:
        for entry in array:
            assert entry == 0
    for array_index, array in enumerate(
        workflowresult[key]["number-op"]["one_body_tensor"]["real"]
    ):
        for entry_index, entry in enumerate(array):
            if array_index == entry_index:
                assert entry == 1
            else:
                assert entry == 0

    for outer_array in workflowresult[key]["number-op"]["two_body_tensor"]["imag"]:
        for middle_array in outer_array:
            for inner_array in middle_array:
                for entry in inner_array:
                    assert entry == 0

print("Workflow result is as expected")
