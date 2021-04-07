import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 1

for key in workflowresult.keys():
    assert workflowresult[key]["class"] == "generate-graph-from-specs"

    assert (
        workflowresult[key]["inputParam:graph-specs"]
        == "{'type_graph': 'regular','num_nodes':2,'degree':1}"
    )

    assert workflowresult[key]["graph"]["schema"] == "zapata-v1-graph"
    assert workflowresult[key]["graph"]["directed"] is False
    assert workflowresult[key]["graph"]["graph"] == {}
    assert workflowresult[key]["graph"]["multigraph"] is False

    assert len(workflowresult[key]["graph"]["links"]) == 1

    assert workflowresult[key]["graph"]["links"][0]["source"] == 0
    assert workflowresult[key]["graph"]["links"][0]["target"] == 1
    assert workflowresult[key]["graph"]["links"][0]["weight"] == 1

    assert len(workflowresult[key]["graph"]["nodes"]) == 2

print("Workflow result is as expected")
