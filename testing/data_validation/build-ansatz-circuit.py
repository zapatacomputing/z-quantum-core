import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 3

found_template = False
found_template_with_params = False
for key in workflowresult.keys():
    if workflowresult[key]["class"] == "build-ansatz-circuit":
        if "inputArtifact:params" not in workflowresult[key].keys():
            found_template = True
            assert (
                workflowresult[key]["inputParam:ansatz-specs"]
                == "{'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockAnsatz', 'number_of_layers': 2, 'problem_size': 2}"
            )
            assert workflowresult[key]["circuit"]["schema"] == "zapata-v1-circuit"

            assert workflowresult[key]["circuit"]["qubits"][0]["index"] == 0
            assert workflowresult[key]["circuit"]["qubits"][1]["index"] == 1

            assert len(workflowresult[key]["circuit"]["gates"]) == 4

            assert workflowresult[key]["circuit"]["gates"][0]["name"] == "Rx"
            assert (
                workflowresult[key]["circuit"]["gates"][0]["params"][0]["params"]
                == "theta_0"
            )
            assert workflowresult[key]["circuit"]["gates"][0]["qubits"][0]["index"] == 0

            assert workflowresult[key]["circuit"]["gates"][1]["name"] == "Rx"
            assert (
                workflowresult[key]["circuit"]["gates"][1]["params"][0]["params"]
                == "theta_0"
            )
            assert workflowresult[key]["circuit"]["gates"][1]["qubits"][0]["index"] == 1

            assert workflowresult[key]["circuit"]["gates"][2]["name"] == "Rx"
            assert (
                workflowresult[key]["circuit"]["gates"][2]["params"][0]["params"]
                == "theta_1"
            )
            assert workflowresult[key]["circuit"]["gates"][2]["qubits"][0]["index"] == 0

            assert workflowresult[key]["circuit"]["gates"][3]["name"] == "Rx"
            assert (
                workflowresult[key]["circuit"]["gates"][3]["params"][0]["params"]
                == "theta_1"
            )
            assert workflowresult[key]["circuit"]["gates"][3]["qubits"][0]["index"] == 1

        if "inputArtifact:params" in workflowresult[key].keys():
            found_template_with_params = True
            assert (
                workflowresult[key]["inputParam:ansatz-specs"]
                == "{'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockAnsatz', 'number_of_layers': 2, 'problem_size': 2}"
            )
            assert workflowresult[key]["circuit"]["schema"] == "zapata-v1-circuit"

            assert workflowresult[key]["circuit"]["qubits"][0]["index"] == 0
            assert workflowresult[key]["circuit"]["qubits"][1]["index"] == 1

            assert len(workflowresult[key]["circuit"]["gates"]) == 4

            assert workflowresult[key]["circuit"]["gates"][0]["name"] == "Rx"
            assert (
                workflowresult[key]["circuit"]["gates"][0]["params"][0]["params"]
                == -1.9372578516205565
            )
            assert workflowresult[key]["circuit"]["gates"][0]["qubits"][0]["index"] == 0

            assert workflowresult[key]["circuit"]["gates"][1]["name"] == "Rx"
            assert (
                workflowresult[key]["circuit"]["gates"][1]["params"][0]["params"]
                == -1.9372578516205565
            )
            assert workflowresult[key]["circuit"]["gates"][1]["qubits"][0]["index"] == 1

            assert workflowresult[key]["circuit"]["gates"][2]["name"] == "Rx"
            assert (
                workflowresult[key]["circuit"]["gates"][2]["params"][0]["params"]
                == 0.7668430821301442
            )
            assert workflowresult[key]["circuit"]["gates"][2]["qubits"][0]["index"] == 0

            assert workflowresult[key]["circuit"]["gates"][3]["name"] == "Rx"
            assert (
                workflowresult[key]["circuit"]["gates"][3]["params"][0]["params"]
                == 0.7668430821301442
            )
            assert workflowresult[key]["circuit"]["gates"][3]["qubits"][0]["index"] == 1

assert found_template is True
assert found_template_with_params is True
print("Workflow result is as expected")
