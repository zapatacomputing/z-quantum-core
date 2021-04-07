import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 1

for key in workflowresult.keys():
    assert workflowresult[key]["class"] == "generate-complete-graph"

    assert workflowresult[key]["inputParam:n-nodes"] == "4"
    assert workflowresult[key]["inputParam:random-weights"] == "false"
    assert workflowresult[key]["inputParam:seed"] == "1234"

    assert workflowresult[key]["graph"]["schema"] == "zapata-v1-graph"

    assert workflowresult[key]["graph"]["directed"] is False
    assert workflowresult[key]["graph"]["graph"] == {}
    assert workflowresult[key]["graph"]["multigraph"] is False

    assert len(workflowresult[key]["graph"]["links"]) == 6

    assert workflowresult[key]["graph"]["links"][0]["source"] == 0
    assert workflowresult[key]["graph"]["links"][0]["target"] == 1
    assert workflowresult[key]["graph"]["links"][0]["weight"] == 0.9664535356921388

    assert workflowresult[key]["graph"]["links"][1]["source"] == 0
    assert workflowresult[key]["graph"]["links"][1]["target"] == 2
    assert workflowresult[key]["graph"]["links"][1]["weight"] == 0.4407325991753527

    assert workflowresult[key]["graph"]["links"][5]["source"] == 2
    assert workflowresult[key]["graph"]["links"][5]["target"] == 3
    assert workflowresult[key]["graph"]["links"][5]["weight"] == 0.5822275730589491

    assert len(workflowresult[key]["graph"]["nodes"]) == 4

print("Workflow result is as expected")
