import pytest
import os
import sys
import numpy as np
from zquantum.core.utils import RNDSEED
from zquantum.core.circuit import load_circuit_template_params

sys.path.append("../..")
from steps.circuit import generate_random_ansatz_params, combine_ansatz_params


class Test_generate_random_ansatz_params:
    @pytest.mark.parametrize(
        "number_of_layers, problem_size",
        [
            (1, 1),
            (0, 1),
            (1, 6),
            (4, 3),
        ],
    )
    def test_generate_random_ansatz_params_using_mock_ansatz_specs(
        self, number_of_layers, problem_size
    ):
        # Given
        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": number_of_layers,
            "problem_size": problem_size,
        }
        seed = RNDSEED

        filename = "params.json"
        if os.path.exists(filename):
            os.remove(filename)

        # When
        generate_random_ansatz_params(ansatz_specs=ansatz_specs, seed=seed)

        # Then
        assert os.path.exists(filename)
        parameters = load_circuit_template_params(filename)
        assert len(parameters) == number_of_layers
        for parameter in parameters:
            assert parameter < np.pi * 0.5
            assert parameter > -np.pi * 0.5
            assert isinstance(parameter, float)

        if os.path.exists(filename):
            os.remove(filename)

    @pytest.mark.parametrize(
        "number_of_parameters",
        [i for i in range(12)],
    )
    def test_generate_random_ansatz_params_using_number_of_parameters(
        self,
        number_of_parameters,
    ):
        # Given
        seed = RNDSEED

        filename = "params.json"
        if os.path.exists(filename):
            os.remove(filename)

        # When
        generate_random_ansatz_params(
            number_of_parameters=number_of_parameters, seed=seed
        )

        # Then
        assert os.path.exists(filename)
        parameters = load_circuit_template_params(filename)
        assert len(parameters) == number_of_parameters
        for parameter in parameters:
            assert parameter < np.pi * 0.5
            assert parameter > -np.pi * 0.5
            assert isinstance(parameter, float)

        if os.path.exists(filename):
            os.remove(filename)

    def test_generate_random_ansatz_params_fails_with_both_ansatz_specs_and_number_of_parameters(
        self,
    ):
        # Given
        number_of_parameters = 2
        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": 2,
            "problem_size": 1,
        }
        seed = RNDSEED

        # When
        with pytest.raises(AssertionError):
            generate_random_ansatz_params(
                ansatz_specs=ansatz_specs,
                number_of_parameters=number_of_parameters,
                seed=seed,
            )


class Test_combine_ansatz_params:
    @pytest.mark.parametrize(
        "params1, params2",
        [
            ([], []),
            ([1.0], []),
            ([], [1.0]),
            ([0.0], [1.0]),
            ([0.0, 1.0, 3.0, 5.0, -2.3], [1.0]),
        ],
    )
    def test_combine_ansatz_params(self, params1, params2):
        # Given
        params1_filename = "params1.json"
        with open(params1_filename, "w") as f:
            f.write(
                '{"schema": "zapata-v1-circuit_template_params","parameters": {"real": '
                + str(params1)
                + "}}"
            )

        params2_filename = "params2.json"
        with open(params2_filename, "w") as f:
            f.write(
                '{"schema": "zapata-v1-circuit_template_params","parameters": {"real": '
                + str(params2)
                + "}}"
            )

        # When
        combine_ansatz_params(params1_filename, params2_filename)

        # Then
        combined_parameters_filename = "combined-params.json"
        assert os.path.exists(combined_parameters_filename)
        parameters = load_circuit_template_params(combined_parameters_filename)
        assert all(parameters == params1 + params2)

        os.remove(params1_filename)
        os.remove(params2_filename)
        os.remove(combined_parameters_filename)