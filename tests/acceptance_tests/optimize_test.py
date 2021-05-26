import json
import os
import shutil

from openfermion import QubitOperator

from steps.optimize import optimize_parametrized_circuit_for_ground_state_of_operator

TARGET_OPERATOR = QubitOperator("X0 X1 Z2 Y4", 1.5)

V1_CIRCUIT_DICT = {
    "schema": "zapata-v1-circuit",
    "name": "Unnamed",
    "gates": [
        {
            "name": "Rx",
            "qubits": [{"index": 0, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_0"],
        },
        {
            "name": "Rx",
            "qubits": [{"index": 1, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_0"],
        },
        {
            "name": "Rx",
            "qubits": [{"index": 2, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_0"],
        },
        {
            "name": "Rx",
            "qubits": [{"index": 3, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_0"],
        },
        {
            "name": "Ry",
            "qubits": [{"index": 0, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_1"],
        },
        {
            "name": "Ry",
            "qubits": [{"index": 1, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_1"],
        },
        {
            "name": "Ry",
            "qubits": [{"index": 2, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_1"],
        },
        {
            "name": "Ry",
            "qubits": [{"index": 3, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_1"],
        },
        {
            "name": "Rz",
            "qubits": [{"index": 0, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_2"],
        },
        {
            "name": "Rz",
            "qubits": [{"index": 1, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_2"],
        },
        {
            "name": "Rz",
            "qubits": [{"index": 2, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_2"],
        },
        {
            "name": "Rz",
            "qubits": [{"index": 3, "info": {"label": "none"}}],
            "info": {"label": "none"},
            "params": ["theta_2"],
        },
    ],
    "qubits": [
        {"index": 0, "info": {"label": "none"}},
        {"index": 1, "info": {"label": "none"}},
        {"index": 2, "info": {"label": "none"}},
        {"index": 3, "info": {"label": "none"}},
    ],
    "info": {"label": None},
}


class TestOptimizeParamterizedCircuit:
    def test_optimizer_specs_input(self):
        circuit_path = "circuit.json"
        with open(circuit_path, "w") as f:
            f.write(json.dumps(V1_CIRCUIT_DICT))

        backend_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockQuantumSimulator",
            "n_samples": 10000,
        }
        initial_parameters = "initial_parameters.json"
        with open(initial_parameters, "w") as f:
            json.dump(
                {
                    "schema": "zapata-v1-circuit_template_params",
                    "parameters": {"real": [1.0, 1.0]},
                },
                f,
            )

        fixed_parameters_path = "fixed_parameters.json"
        with open(fixed_parameters_path, "w") as f:
            json.dump(
                {
                    "schema": "zapata-v1-circuit_template_params",
                    "parameters": {"real": [1.0]},
                },
                f,
            )
        estimation_method_specs = json.dumps(
            {
                "module_name": "zquantum.core.estimation",
                "function_name": "estimate_expectation_values_by_averaging",
            }
        )
        estimation_preprocessors_specs = [
            json.dumps(
                {
                    "module_name": "zquantum.core.estimation",
                    "function_name": "group_greedily",
                }
            ),
            json.dumps(
                {
                    "module_name": "zquantum.core.estimation",
                    "function_name": "perform_context_selection",
                }
            ),
            json.dumps(
                {
                    "module_name": "zquantum.core.estimation",
                    "function_name": "allocate_shots_uniformly",
                    "number_of_shots": 10000,
                }
            ),
        ]
        optimization_results_path = "optimization-results.json"
        shutil.rmtree(optimization_results_path, ignore_errors=True)

        optimizer_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockOptimizer",
        }

        optimize_parametrized_circuit_for_ground_state_of_operator(
            optimizer_specs=json.dumps(optimizer_specs),
            target_operator=TARGET_OPERATOR,
            circuit=circuit_path,
            backend_specs=json.dumps(backend_specs),
            estimation_method_specs=json.dumps(estimation_method_specs),
            estimation_preprocessors_specs=estimation_preprocessors_specs,
            initial_parameters=initial_parameters,
            fixed_parameters=fixed_parameters_path,
            parameter_precision=0.001,
            parameter_precision_seed=1234,
        )

        assert os.path.exists(optimization_results_path)
        assert os.path.exists("optimized-parameters.json")

        os.remove(optimization_results_path)
        os.remove("optimized-parameters.json")
        os.remove(circuit_path)
        os.remove("initial_parameters.json")
        os.remove(fixed_parameters_path)
