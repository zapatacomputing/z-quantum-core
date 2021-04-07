import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 4

found_template = False
for key in workflowresult.keys():
    if workflowresult[key]["class"] == "get-expectation-values-for-qubit-operator":
        found_template = True
        assert (
            workflowresult[key]["inputParam:backend-specs"]
            == "{'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumBackend', 'n_samples': 1000}"
        )

        assert (
            workflowresult[key]["expectation-values"]["schema"]
            == "zapata-v1-expectation_values"
        )

        identity_term = workflowresult[key]["expectation-values"]["expectation_values"][
            "real"
        ][0]
        assert isinstance(float(identity_term), float)
        assert identity_term > 1.5
        assert identity_term < 2.5

        for expectation_value in workflowresult[key]["expectation-values"][
            "expectation_values"
        ]["real"][1:]:
            assert isinstance(float(expectation_value), float)
            assert expectation_value > -0.5
            assert expectation_value < 0.5

assert found_template
print("Workflow result is as expected")
