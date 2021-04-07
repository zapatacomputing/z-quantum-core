import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 2

found_template = False
found_template_with_ansatz_specs = False

for key in workflowresult.keys():
    if workflowresult[key]["class"] == "build-uniform-parameter-grid":
        if workflowresult[key]["inputParam:ansatz-specs"] == "None":
            found_template = True
            assert workflowresult[key]["inputParam:number-of-params-per-layer"] == "2"
            assert workflowresult[key]["inputParam:n-layers"] == "2"
            assert workflowresult[key]["inputParam:min-value"] == "-3.14"
            assert workflowresult[key]["inputParam:max-value"] == "3.14"
            assert workflowresult[key]["inputParam:step"] == ".314"
            assert (
                workflowresult[key]["parameter-grid"]["schema"]
                == "zapata-v1-parameter_grid"
            )

            assert len(workflowresult[key]["parameter-grid"]["param_ranges"]) == 4

            for param_range in workflowresult[key]["parameter-grid"]["param_ranges"]:
                assert param_range["param_ranges"][0]["param_ranges"] == -3.14
                assert param_range["param_ranges"][1]["param_ranges"] == 3.14
                assert param_range["param_ranges"][2]["param_ranges"] == 0.314

        elif workflowresult[key]["inputParam:ansatz-specs"] != "None":
            found_template_with_ansatz_specs = True
            assert (
                workflowresult[key]["inputParam:number-of-params-per-layer"] == "None"
            )
            assert (
                workflowresult[key]["inputParam:ansatz-specs"]
                == "{'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockAnsatz', 'number_of_layers': 2, 'problem_size': 2}"
            )
            assert workflowresult[key]["inputParam:n-layers"] == "2"
            assert workflowresult[key]["inputParam:min-value"] == "-3.14"
            assert workflowresult[key]["inputParam:max-value"] == "3.14"
            assert workflowresult[key]["inputParam:step"] == ".314"
            assert (
                workflowresult[key]["parameter-grid"]["schema"]
                == "zapata-v1-parameter_grid"
            )

            assert len(workflowresult[key]["parameter-grid"]["param_ranges"]) == 4

            for param_range in workflowresult[key]["parameter-grid"]["param_ranges"]:
                assert param_range["param_ranges"][0]["param_ranges"] == -3.14
                assert param_range["param_ranges"][1]["param_ranges"] == 3.14
                assert param_range["param_ranges"][2]["param_ranges"] == 0.314

assert found_template is True
assert found_template_with_ansatz_specs is True
print("Workflow result is as expected")
