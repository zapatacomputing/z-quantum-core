import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 3

found_template = False
for key in workflowresult.keys():
    if workflowresult[key]["class"] == "get-ground-state-at-particle-number-jw":
        found_template = True
        assert workflowresult[key]["inputParam:particle-number"] == "2"

        assert workflowresult[key]["ground-state"]["schema"] == "zapata-v1-wavefunction"

        assert len(workflowresult[key]["ground-state"]["amplitudes"]["real"]) == 16
        for element in workflowresult[key]["ground-state"]["amplitudes"]["real"]:
            assert isinstance(float(element), float)
            assert element >= -1
            assert element <= 1

        assert len(workflowresult[key]["ground-state"]["amplitudes"]["imag"]) == 16
        for element in workflowresult[key]["ground-state"]["amplitudes"]["imag"]:
            assert isinstance(float(element), float)
            assert element >= -1
            assert element <= 1

        assert (
            workflowresult[key]["value-estimate"]["schema"]
            == "zapata-v1-value_estimate"
        )
        assert workflowresult[key]["value-estimate"]["precision"] is None
        assert workflowresult[key]["value-estimate"]["value"] == 2


assert found_template
print("Workflow result is as expected")
