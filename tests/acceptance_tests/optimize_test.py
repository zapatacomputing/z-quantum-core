import json
import os

from openfermion import QubitOperator

from steps.optimize import optimize_parametrized_circuit_for_ground_state_of_operator

TARGET_OPERATOR = QubitOperator("X0 X1 Z2 Y4", 1.5)


class TestOptimizeParamterizedCircuit:
    def test_optimizer_specs_input(self):
        optimizer_specs = (
            '{"module_name": "zquantum.core.interfaces.mock_objects", '
            '"function_name": "MockOptimizer"}'
        )
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
        backend_specs = (
            '{"module_name": "zquantum.core.interfaces.mock_objects", '
            '"function_name": "MockQuantumSimulator", "n_samples": 10000}'
        )
        initial_parameters = "initial_parameters.json"
        with open(initial_parameters, "w") as f:
            f.write(
                '{"schema": "zapata-v1-circuit_template_params","parameters": {"real": '
                "[1.0, 1.0]}}"
            )
        fixed_parameters = "fixed_parameters.json"
        with open(fixed_parameters, "w") as f:
            f.write(
                '{"schema": "zapata-v1-circuit_template_params","parameters": {"real": '
                "[1.0]}}"
            )

        estimation_method_specs = (
            '{"module_name": "zquantum.core.estimation", '
            '"function_name": "estimate_expectation_values_by_averaging"}'
        )
        estimation_preprocessors_specs = [
            (
                '{"module_name": "zquantum.core.estimation", "function_name": '
                '"group_greedily"}',
            ),
            (
                '{"module_name": "zquantum.core.estimation", "function_name": '
                '"perform_context_selection"}'
            ),
            (
                '{"module_name": "zquantum.core.estimation", "function_name": '
                '"allocate_shots_uniformly", "number_of_shots": 10000}'
            ),
        ]
        if os.path.exists("optimization_results.json"):
            os.remove("optimization_results.json")

        optimize_parametrized_circuit_for_ground_state_of_operator(
            optimizer_specs,
            target_operator,
            circuit,
            backend_specs,
            estimation_method_specs,
            estimation_preprocessors_specs,
            initial_parameters=initial_parameters,
            fixed_parameters=fixed_parameters,
            parameter_precision=0.001,
            parameter_precision_seed=1234,
        )
        assert os.path.exists("optimization-results.json")
        assert os.path.exists("optimized-parameters.json")
        os.remove("optimization-results.json")
        os.remove("optimized-parameters.json")
        os.remove("circuit.json")
        os.remove("initial_parameters.json")
        os.remove("fixed_parameters.json")
