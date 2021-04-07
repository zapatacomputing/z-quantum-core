import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 2

found_template = False
for key in workflowresult.keys():
    if workflowresult[key]["class"] == "transform-interaction-operator":
        found_template = True
        assert workflowresult[key]["inputParam:transformation"] == "Jordan-Wigner"

        assert workflowresult[key]["timing"]["schema"] == "zapata-v1-timing"
        assert workflowresult[key]["timing"]["walltime"] > 0.0

        assert workflowresult[key]["transformed-op"]["schema"] == "zapata-v1-qubit_op"
        assert len(workflowresult[key]["transformed-op"]["terms"]) == 5

        assert (
            workflowresult[key]["transformed-op"]["terms"][0]["coefficient"]["real"]
            == 2
        )
        assert (
            workflowresult[key]["transformed-op"]["terms"][0]["coefficient"]["imag"]
            == 0
        )

        assert (
            workflowresult[key]["transformed-op"]["terms"][1]["coefficient"]["real"]
            == -0.5
        )
        assert (
            workflowresult[key]["transformed-op"]["terms"][1]["coefficient"]["imag"]
            == 0
        )
        assert len(workflowresult[key]["transformed-op"]["terms"][1]["pauli_ops"]) == 1
        assert (
            workflowresult[key]["transformed-op"]["terms"][1]["pauli_ops"][0]["op"]
            == "Z"
        )
        assert (
            workflowresult[key]["transformed-op"]["terms"][1]["pauli_ops"][0]["qubit"]
            == 0
        )

        assert (
            workflowresult[key]["transformed-op"]["terms"][2]["coefficient"]["real"]
            == -0.5
        )
        assert (
            workflowresult[key]["transformed-op"]["terms"][2]["coefficient"]["imag"]
            == 0
        )
        assert len(workflowresult[key]["transformed-op"]["terms"][2]["pauli_ops"]) == 1
        assert (
            workflowresult[key]["transformed-op"]["terms"][2]["pauli_ops"][0]["op"]
            == "Z"
        )
        assert (
            workflowresult[key]["transformed-op"]["terms"][2]["pauli_ops"][0]["qubit"]
            == 1
        )

        assert (
            workflowresult[key]["transformed-op"]["terms"][3]["coefficient"]["real"]
            == -0.5
        )
        assert (
            workflowresult[key]["transformed-op"]["terms"][3]["coefficient"]["imag"]
            == 0
        )
        assert len(workflowresult[key]["transformed-op"]["terms"][3]["pauli_ops"]) == 1
        assert (
            workflowresult[key]["transformed-op"]["terms"][3]["pauli_ops"][0]["op"]
            == "Z"
        )
        assert (
            workflowresult[key]["transformed-op"]["terms"][3]["pauli_ops"][0]["qubit"]
            == 2
        )

        assert (
            workflowresult[key]["transformed-op"]["terms"][4]["coefficient"]["real"]
            == -0.5
        )
        assert (
            workflowresult[key]["transformed-op"]["terms"][4]["coefficient"]["imag"]
            == 0
        )
        assert len(workflowresult[key]["transformed-op"]["terms"][4]["pauli_ops"]) == 1
        assert (
            workflowresult[key]["transformed-op"]["terms"][4]["pauli_ops"][0]["op"]
            == "Z"
        )
        assert (
            workflowresult[key]["transformed-op"]["terms"][4]["pauli_ops"][0]["qubit"]
            == 3
        )

assert found_template
print("Workflow result is as expected")
