import json
import sys

with open(sys.argv[1], "r") as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 5

found_template = False
for key in workflowresult.keys():

    if workflowresult[key]["class"] == "interpolate-qubit-operators":
        found_template = True
        assert workflowresult[key]["inputParam:epsilon"] == "0.5"

        assert (
            workflowresult[key]["output-qubit-operator"]["schema"]
            == "zapata-v1-qubit_op"
        )

        assert len(workflowresult[key]["output-qubit-operator"]["terms"]) == 5

        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][0]["coefficient"][
                "real"
            ]
            == -0.5
        )
        assert (
            len(workflowresult[key]["output-qubit-operator"]["terms"][0]["pauli_ops"])
            == 1
        )
        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][0]["pauli_ops"][0][
                "op"
            ]
            == "Z"
        )
        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][0]["pauli_ops"][0][
                "qubit"
            ]
            == 0
        )

        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][1]["coefficient"][
                "real"
            ]
            == -0.5
        )
        assert (
            len(workflowresult[key]["output-qubit-operator"]["terms"][1]["pauli_ops"])
            == 1
        )
        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][1]["pauli_ops"][0][
                "op"
            ]
            == "Z"
        )
        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][1]["pauli_ops"][0][
                "qubit"
            ]
            == 1
        )

        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][2]["coefficient"][
                "real"
            ]
            == -0.5
        )
        assert (
            len(workflowresult[key]["output-qubit-operator"]["terms"][2]["pauli_ops"])
            == 1
        )
        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][2]["pauli_ops"][0][
                "op"
            ]
            == "Z"
        )
        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][2]["pauli_ops"][0][
                "qubit"
            ]
            == 2
        )

        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][3]["coefficient"][
                "real"
            ]
            == -0.5
        )
        assert (
            len(workflowresult[key]["output-qubit-operator"]["terms"][3]["pauli_ops"])
            == 1
        )
        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][3]["pauli_ops"][0][
                "op"
            ]
            == "Z"
        )
        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][3]["pauli_ops"][0][
                "qubit"
            ]
            == 3
        )

        assert (
            workflowresult[key]["output-qubit-operator"]["terms"][4]["coefficient"][
                "real"
            ]
            == 1
        )

assert found_template
print("Workflow result is as expected")
