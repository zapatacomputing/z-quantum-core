"""Test cases for storing artifacts and recording functions that store them."""
import numpy as np
import pytest
from ..interfaces.functions import CallableStoringArtifacts
from .recorder import recorder, ArtifactCollection, store_artifact
from .save_conditions import every_nth
from .example_functions import function_3, function_4, Function5


def test_store_artifact_by_default_does_not_force_artifacts_storage():
    artifacts = ArtifactCollection()
    store_artifact(artifacts)("bitstring", "1111")
    assert not artifacts.forced


@pytest.mark.parametrize(
    "source_function,params_sequence,expected_artifacts",
    [
        (
            function_3,
            [3, 4, 5],
            [{"bitstring": string} for string in ["11", "100", "101"]],
        ),
        (
            function_4,
            [0, 10, 21],
            [{"bitstring": string} for string in ["0", "1010", "10101"]],
        ),
        (
            Function5(2),
            [np.array([1.5, 2, 3]), np.array([4.0, 2.5, 3.0])],
            [{"something": 3.5}, {"something": 6.5}],
        ),
    ],
)
def test_recorder_stores_all_artifacts_by_default(
    source_function: CallableStoringArtifacts, params_sequence, expected_artifacts
):
    function = recorder(source_function)
    for param in params_sequence:
        function(param)

    assert [entry.artifacts for entry in function.history] == expected_artifacts


def test_recorder_stores_history_entry_if_artifact_is_force_stored():
    function = recorder(function_4, save_condition=every_nth(5))
    for n in [0, 2, 3, 5, 7, 5]:
        function(n)

    assert [entry.call_number for entry in function.history] == [0, 1, 5]
    assert [entry.value for entry in function.history] == [0, 4, 10]
    assert [entry.params for entry in function.history] == [0, 2, 5]
    assert [entry.artifacts for entry in function.history] == [
        {"bitstring": "0"},
        {"bitstring": "10"},
        {"bitstring": "101"},
    ]
