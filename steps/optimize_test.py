import pytest
from optimize import optimize_parameterized_quantum_circuit
import os
import json
from openfermion import QubitOperator
import sympy
import numpy as np

TARGET_OPERATOR = QubitOperator("X0 X1 Z2 Y4", 1.5)


def test_optimize_parameterized_quantum_circuit_optimizer_specs_input():
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
    estimator_specs = "None"
    epsilon = "None"
    delta = "None"
    initial_parameters = "initial_parameters.json"
    with open(initial_parameters, "w") as f:
        f.write(
            '{"schema": "zapata-v1-circuit_template_params","parameters": {"real": [1.0, 1.0, 1.0]}}'
        )

    if os.path.exists("optimization_results.json"):
        os.remove("optimization_results.json")
    optimize_parameterized_quantum_circuit(
        optimizer_specs,
        target_operator,
        circuit,
        backend_specs,
        estimator_specs,
        epsilon,
        delta,
        initial_parameters,
    )
    assert os.path.exists("optimization_results.json")
    os.remove("optimization_results.json")
    os.remove("circuit.json")
    os.remove("initial_parameters.json")
