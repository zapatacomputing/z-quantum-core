import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 1

for key in workflowresult.keys():
    assert workflowresult[key]["class"] == "build-circuit-layers-and-connectivity"
    assert workflowresult[key]["inputParam:layer-type"] == "nearest-neighbor"
    assert workflowresult[key]["inputParam:x-dimension"] == "2"
    assert workflowresult[key]["inputParam:y-dimension"] == "None"

    assert len(workflowresult[key]["circuit-connectivity"]["connectivity"]) == 1
    assert (
        workflowresult[key]["circuit-connectivity"]["schema"]
        == "zapata-v1-circuit_connectivity"
    )
    assert (
        len(
            workflowresult[key]["circuit-connectivity"]["connectivity"][0][
                "connectivity"
            ]
        )
        == 2
    )
    assert (
        workflowresult[key]["circuit-connectivity"]["connectivity"][0]["connectivity"][
            0
        ]["connectivity"]
        == 0
    )
    assert (
        workflowresult[key]["circuit-connectivity"]["connectivity"][0]["connectivity"][
            1
        ]["connectivity"]
        == 1
    )

    assert len(workflowresult[key]["circuit-layers"]["layers"]) == 2
    assert workflowresult[key]["circuit-layers"]["schema"] == "zapata-v1-circuit_layers"
    assert len(workflowresult[key]["circuit-layers"]["layers"][0]["layers"]) == 1
    assert (
        workflowresult[key]["circuit-layers"]["layers"][0]["layers"][0]["layers"][0][
            "layers"
        ]
        == 0
    )
    assert (
        workflowresult[key]["circuit-layers"]["layers"][0]["layers"][0]["layers"][1][
            "layers"
        ]
        == 1
    )

print("Workflow result is as expected")
