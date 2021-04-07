import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 2

for key in workflowresult.keys():
    assert workflowresult[key]["class"] == "generate-random-ansatz-params"

    if workflowresult[key]["inputParam:ansatz-specs"] != "None":
        assert (
            workflowresult[key]["inputParam:ansatz-specs"]
            == "{'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockAnsatz', 'number_of_layers': 2, 'problem_size': 2}"
        )
        assert workflowresult[key]["inputParam:number-of-params"] == "None"
        assert workflowresult[key]["inputParam:min-val"] == "-3.14"
        assert workflowresult[key]["inputParam:max-val"] == "3.14"
        assert workflowresult[key]["inputParam:seed"] == "1234"

        assert (
            workflowresult[key]["params"]["schema"]
            == "zapata-v1-circuit_template_params"
        )

        assert (
            workflowresult[key]["params"]["parameters"]["real"][0]
            == -1.9372578516205565
        )
        assert (
            workflowresult[key]["params"]["parameters"]["real"][1] == 0.7668430821301442
        )

    if workflowresult[key]["inputParam:number-of-params"] != "None":
        assert workflowresult[key]["inputParam:ansatz-specs"] == "None"
        assert workflowresult[key]["inputParam:number-of-params"] == "4"
        assert workflowresult[key]["inputParam:min-val"] == "-3.14"
        assert workflowresult[key]["inputParam:max-val"] == "3.14"
        assert workflowresult[key]["inputParam:seed"] == "1234"

        assert (
            workflowresult[key]["params"]["schema"]
            == "zapata-v1-circuit_template_params"
        )

        assert (
            workflowresult[key]["params"]["parameters"]["real"][0]
            == -1.9372578516205565
        )
        assert (
            workflowresult[key]["params"]["parameters"]["real"][1] == 0.7668430821301442
        )
        assert (
            workflowresult[key]["params"]["parameters"]["real"][2]
            == -0.3910697990353209
        )
        assert (
            workflowresult[key]["params"]["parameters"]["real"][3] == 1.7920519057224706
        )

print("Workflow result is as expected")
