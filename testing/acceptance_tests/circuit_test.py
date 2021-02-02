import pytest
import os
import sys
import numpy as np
from zquantum.core.utils import RNDSEED
from zquantum.core.circuit import load_circuit_template_params

sys.path.append("../..")
from steps.circuit import generate_random_ansatz_params


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
