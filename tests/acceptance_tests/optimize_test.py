import json
import os
import shutil

from openfermion import QubitOperator

from steps.optimize import optimize_parametrized_circuit_for_ground_state_of_operator

TARGET_OPERATOR = QubitOperator("X0 X1 Z2 Y4", 1.5)


# To regenerate the circuit, run:
# import zquantum.core.circuits as new_circuits
# import sympy
# print(
#     new_circuits.to_dict(
#         new_circuits.Circuit(
#             [
#                 *[new_circuits.RX(sympy.Symbol("theta_0"))(i) for i in range(4)],
#                 *[new_circuits.RY(sympy.Symbol("theta_1"))(i) for i in range(4)],
#                 *[new_circuits.RZ(sympy.Symbol("theta_2"))(i) for i in range(4)],
#             ]
#         )
#     )
# )
CIRCUIT_DICT = {
    "schema": "zapata-v1-circuit-v2",
    "n_qubits": 4,
    "operations": [
        {
            "type": "gate_operation",
            "gate": {"name": "RX", "params": ["theta_0"], "free_symbols": ["theta_0"]},
            "qubit_indices": [0],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RX", "params": ["theta_0"], "free_symbols": ["theta_0"]},
            "qubit_indices": [1],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RX", "params": ["theta_0"], "free_symbols": ["theta_0"]},
            "qubit_indices": [2],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RX", "params": ["theta_0"], "free_symbols": ["theta_0"]},
            "qubit_indices": [3],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RY", "params": ["theta_1"], "free_symbols": ["theta_1"]},
            "qubit_indices": [0],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RY", "params": ["theta_1"], "free_symbols": ["theta_1"]},
            "qubit_indices": [1],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RY", "params": ["theta_1"], "free_symbols": ["theta_1"]},
            "qubit_indices": [2],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RY", "params": ["theta_1"], "free_symbols": ["theta_1"]},
            "qubit_indices": [3],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RZ", "params": ["theta_2"], "free_symbols": ["theta_2"]},
            "qubit_indices": [0],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RZ", "params": ["theta_2"], "free_symbols": ["theta_2"]},
            "qubit_indices": [1],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RZ", "params": ["theta_2"], "free_symbols": ["theta_2"]},
            "qubit_indices": [2],
        },
        {
            "type": "gate_operation",
            "gate": {"name": "RZ", "params": ["theta_2"], "free_symbols": ["theta_2"]},
            "qubit_indices": [3],
        },
    ],
}


class TestOptimizeParamterizedCircuit:
    def test_optimizer_specs_input(self):
        circuit_path = "circuit.json"
        with open(circuit_path, "w") as f:
            json.dump(CIRCUIT_DICT, f)

        backend_specs = {
            "module_name": "zquantum.core.symbolic_simulator",
            "function_name": "SymbolicSimulator",
            "n_samples": 10000,
        }
        initial_parameters_path = "initial_parameters.json"
        with open(initial_parameters_path, "w") as f:
            json.dump(
                {
                    "schema": "zapata-v1-array",
                    "array": {"real": [1.0, 1.0]},
                },
                f,
            )

        fixed_parameters_path = "fixed_parameters.json"
        with open(fixed_parameters_path, "w") as f:
            json.dump(
                {
                    "schema": "zapata-v1-array",
                    "array": {"real": [1.0]},
                },
                f,
            )
        estimation_method_specs = {
            "module_name": "zquantum.core.estimation",
            "function_name": "estimate_expectation_values_by_averaging",
        }
        estimation_preprocessors_specs = [
            {
                "module_name": "zquantum.core.estimation",
                "function_name": "group_greedily",
            },
            {
                "module_name": "zquantum.core.estimation",
                "function_name": "perform_context_selection",
            },
            {
                "module_name": "zquantum.core.estimation",
                "function_name": "allocate_shots_uniformly",
                "number_of_shots": 10000,
            },
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
            parametrized_circuit=circuit_path,
            backend_specs=json.dumps(backend_specs),
            estimation_method_specs=json.dumps(estimation_method_specs),
            estimation_preprocessors_specs=[
                json.dumps(spec) for spec in estimation_preprocessors_specs
            ],
            initial_parameters=initial_parameters_path,
            fixed_parameters=fixed_parameters_path,
            parameter_precision=0.001,
            parameter_precision_seed=1234,
        )

        assert os.path.exists(optimization_results_path)
        assert os.path.exists("optimized-parameters.json")

        os.remove(optimization_results_path)
        os.remove("optimized-parameters.json")
        os.remove(circuit_path)
        os.remove(initial_parameters_path)
        os.remove(fixed_parameters_path)
