import json
import sys

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 4

found_template = False
for key in workflowresult.keys():
    if workflowresult[key]["class"] == "evaluate-operator-for-parameter-grid":
        found_template = True
        assert (
            workflowresult[key]["inputParam:backend-specs"]
            == "{'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumBackend', 'n_samples': 1000}"
        )
        assert (
            workflowresult[key]["inputParam:ansatz-specs"]
            == "{'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockAnsatz', 'number_of_layers': 2, 'problem_size': 2}"
        )
        assert workflowresult[key]["inputParam:previous-layer-parameters"] == "[]"

        assert (
            workflowresult[key]["optimal-parameters"]["schema"]
            == "zapata-v1-circuit_template_params"
        )
        assert len(workflowresult[key]["optimal-parameters"]["parameters"]["real"]) == 2
        assert isinstance(
            float(workflowresult[key]["optimal-parameters"]["parameters"]["real"][0]),
            float,
        )
        assert isinstance(
            float(workflowresult[key]["optimal-parameters"]["parameters"]["real"][1]),
            float,
        )

        assert (
            workflowresult[key]["parameter-grid-evaluation"]["schema"]
            == "zapata-v1-parameter_grid_evaluation"
        )

        assert (
            len(workflowresult[key]["parameter-grid-evaluation"]["values_set"]) == 400
        )

        for evaluation in workflowresult[key]["parameter-grid-evaluation"][
            "values_set"
        ]:
            assert evaluation["parameter1"] >= -3.14
            assert evaluation["parameter1"] <= 3.14
            assert evaluation["parameter2"] >= -3.14
            assert evaluation["parameter2"] <= 3.14

            assert evaluation["value"]["precision"] is None
            assert evaluation["value"]["schema"] == "zapata-v1-value_estimate"
            assert isinstance(float(evaluation["value"]["value"]), float)


assert found_template
print("Workflow result is as expected")
