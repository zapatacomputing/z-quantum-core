import json
import os
import sys

from openfermion import QubitOperator

sys.path.append("../..")
from steps.optimize import optimize_parametrized_circuit_for_ground_state_of_operator

TARGET_OPERATOR = QubitOperator("X0 X1 Z2 Y4", 1.5)


def test_optimize_parametrized_circuit_for_ground_state_of_operator_optimizer_specs_input():
    optimizer_specs = '{"module_name": "zquantum.core.interfaces.mock_objects", "function_name": "MockOptimizer"}'
    target_operator = TARGET_OPERATOR
    circuit = "circuit.json"
    with open(circuit, "w") as f:
        f.write(
            json.dumps(
                {
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
            )
        )
    backend_specs = '{"module_name": "zquantum.core.interfaces.mock_objects", "function_name": "MockQuantumSimulator", "n_samples": 10000}'
    initial_parameters = "initial_parameters.json"
    with open(initial_parameters, "w") as f:
        f.write(
            '{"schema": "zapata-v1-circuit_template_params","parameters": {"real": [1.0, 1.0]}}'
        )
    fixed_parameters = "fixed_parameters.json"
    with open(fixed_parameters, "w") as f:
        f.write(
            '{"schema": "zapata-v1-circuit_template_params","parameters": {"real": [1.0]}}'
        )

    if os.path.exists("optimization_results.json"):
        os.remove("optimization_results.json")
    optimize_parametrized_circuit_for_ground_state_of_operator(
        optimizer_specs,
        target_operator,
        circuit,
        backend_specs,
        initial_parameters=initial_parameters,
        fixed_parameters=fixed_parameters,
        parameter_precision=0.001,
        parameter_precision_seed=1234,
    )
    assert os.path.exists("optimization_results.json")
    os.remove("optimization_results.json")
    os.remove("circuit.json")
    os.remove("initial_parameters.json")
    os.remove("fixed_parameters.json")
