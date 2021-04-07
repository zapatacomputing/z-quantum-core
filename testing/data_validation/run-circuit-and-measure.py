import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 2

found_template = False
for key in workflowresult.keys():
    if workflowresult[key]["class"] == "run-circuit-and-measure":
        found_template = True
        assert (
            workflowresult[key]["inputParam:backend-specs"]
            == "{'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumBackend', 'n_samples': 1000}"
        )
        assert workflowresult[key]["measurements"]["schema"] == "zapata-v1-measurements"

        assert len(workflowresult[key]["measurements"]["bitstrings"]) == 1000

        assert len(workflowresult[key]["measurements"]["counts"].keys()) == 16
        total_num_bitstrings = 0
        for bitstring in workflowresult[key]["measurements"]["counts"].keys():
            assert len(bitstring) == 4
            total_num_bitstrings += workflowresult[key]["measurements"]["counts"][
                bitstring
            ]
        assert total_num_bitstrings == 1000

assert found_template is True
print("Workflow result is as expected")
