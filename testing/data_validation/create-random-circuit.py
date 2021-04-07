import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 1

for key in workflowresult.keys():
    assert workflowresult[key]["class"] == "create-random-circuit"
    assert workflowresult[key]["inputParam:num-qubits"] == "4"
    assert workflowresult[key]["inputParam:num-gates"] == "20"
    assert workflowresult[key]["inputParam:seed"] == "1234"

    assert workflowresult[key]["circuit"]["schema"] == "zapata-v1-circuit"

    assert workflowresult[key]["circuit"]["qubits"][0]["index"] == 0
    assert workflowresult[key]["circuit"]["qubits"][1]["index"] == 2
    assert workflowresult[key]["circuit"]["qubits"][2]["index"] == 1
    assert workflowresult[key]["circuit"]["qubits"][3]["index"] == 3

    assert workflowresult[key]["circuit"]["gates"][0]["name"] == "CPHASE"
    assert (
        workflowresult[key]["circuit"]["gates"][0]["params"][0]["params"]
        == -1.6536729524317533
    )
    assert workflowresult[key]["circuit"]["gates"][0]["qubits"][0]["index"] == 0
    assert workflowresult[key]["circuit"]["gates"][0]["qubits"][1]["index"] == 2

    assert workflowresult[key]["circuit"]["gates"][4]["name"] == "CNOT"
    assert workflowresult[key]["circuit"]["gates"][4]["qubits"][0]["index"] == 3
    assert workflowresult[key]["circuit"]["gates"][4]["qubits"][1]["index"] == 0

    assert workflowresult[key]["circuit"]["gates"][9]["name"] == "SWAP"
    assert workflowresult[key]["circuit"]["gates"][9]["qubits"][0]["index"] == 3
    assert workflowresult[key]["circuit"]["gates"][9]["qubits"][1]["index"] == 1

    assert workflowresult[key]["circuit"]["gates"][14]["name"] == "Y"
    assert workflowresult[key]["circuit"]["gates"][14]["qubits"][0]["index"] == 2

    assert workflowresult[key]["circuit"]["gates"][19]["name"] == "CZ"
    assert workflowresult[key]["circuit"]["gates"][19]["qubits"][0]["index"] == 3
    assert workflowresult[key]["circuit"]["gates"][19]["qubits"][1]["index"] == 2

print("Workflow result is as expected")
