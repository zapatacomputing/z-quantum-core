import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 3

found_correct_template = False
for key in workflowresult.keys():
    if workflowresult[key]["class"] == "combine-ansatz-params":
        found_correct_template = True

        assert (
            workflowresult[key]["combined-params"]["schema"]
            == "zapata-v1-circuit_template_params"
        )

        assert len(workflowresult[key]["combined-params"]["parameters"]["real"]) == 3

        assert (
            workflowresult[key]["combined-params"]["parameters"]["real"][0]
            == -1.9372578516205565
        )
        assert (
            workflowresult[key]["combined-params"]["parameters"]["real"][1]
            == 0.7668430821301442
        )
        assert (
            workflowresult[key]["combined-params"]["parameters"]["real"][2]
            == -1.9372578516205565
        )

assert found_correct_template
print("Workflow result is as expected")
