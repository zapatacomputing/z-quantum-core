import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 2

found_template = False
for key in workflowresult.keys():
    if workflowresult[key]["class"] == "get-diagonal-component":
        found_template = True

        assert (
            workflowresult[key]["diagonal-op"]["schema"] == "zapata-v1-interaction_op"
        )
        assert workflowresult[key]["diagonal-op"]["constant"]["real"] == 0
        for row in workflowresult[key]["diagonal-op"]["one_body_tensor"]["imag"]:
            for element in row:
                assert element == 0
        for row_index, row in enumerate(
            workflowresult[key]["diagonal-op"]["one_body_tensor"]["real"]
        ):
            for element_index, element in enumerate(row):
                if row_index == element_index:
                    assert element == 1
                else:
                    assert element == 0

        for first_layer in workflowresult[key]["diagonal-op"]["two_body_tensor"][
            "imag"
        ]:
            for second_layer in first_layer:
                for third_layer in second_layer:
                    for element in third_layer:
                        assert element == 0
        for first_layer in workflowresult[key]["diagonal-op"]["two_body_tensor"][
            "real"
        ]:
            for second_layer in first_layer:
                for third_layer in second_layer:
                    for element in third_layer:
                        assert element == 0

        assert (
            workflowresult[key]["remainder-op"]["schema"] == "zapata-v1-interaction_op"
        )
        assert workflowresult[key]["remainder-op"]["constant"]["real"] == 0
        for row in workflowresult[key]["remainder-op"]["one_body_tensor"]["imag"]:
            for element in row:
                assert element == 0
        for row in workflowresult[key]["remainder-op"]["one_body_tensor"]["real"]:
            for element in row:
                assert element == 0

        for first_layer in workflowresult[key]["remainder-op"]["two_body_tensor"][
            "imag"
        ]:
            for second_layer in first_layer:
                for third_layer in second_layer:
                    for element in third_layer:
                        assert element == 0
        for first_layer in workflowresult[key]["remainder-op"]["two_body_tensor"][
            "real"
        ]:
            for second_layer in first_layer:
                for third_layer in second_layer:
                    for element in third_layer:
                        assert element == 0


assert found_template
print("Workflow result is as expected")
