import json
import sys

import numpy as np

with open(sys.argv[1]) as f:
    workflowresult = json.loads(f.read())

assert len(workflowresult.keys()) == 2

found_template = False
for key in workflowresult.keys():
    if workflowresult[key]["class"] == "get-bitstring-distribution":
        found_template = True
        assert (
            workflowresult[key]["inputParam:backend-specs"]
            == "{'module_name': 'zquantum.core.interfaces.mock_objects', 'function_name': 'MockQuantumBackend', 'n_samples': 100}"
        )
        assert (
            workflowresult[key]["bitstring-distribution"]["schema"]
            == "zapata-v1-bitstring-probability-distribution"
        )

        assert (
            len(
                workflowresult[key]["bitstring-distribution"][
                    "bitstring_distribution"
                ].keys()
            )
            == 16
        )
        total_probability = 0
        for bitstring in workflowresult[key]["bitstring-distribution"][
            "bitstring_distribution"
        ].keys():
            assert len(bitstring) == 4
            assert (
                workflowresult[key]["bitstring-distribution"]["bitstring_distribution"][
                    bitstring
                ]
                >= 0.0
            )

            assert (
                workflowresult[key]["bitstring-distribution"]["bitstring_distribution"][
                    bitstring
                ]
                <= 1.0
            )
            total_probability += workflowresult[key]["bitstring-distribution"][
                "bitstring_distribution"
            ][bitstring]

        print(total_probability)
        assert np.isclose(total_probability, 1.0)

assert found_template is True
print("Workflow result is as expected")
