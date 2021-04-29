import pytest
import sympy
from sympy.core.symbol import Symbol
from zquantum.core.wip.circuits import MultiPhaseOperation
import numpy as np


class TestMultiPhaseOperation:
    @pytest.mark.parametrize(
        "wavefunction",
        [
            np.array([1, 0, 0, 0]),
            np.array([1, 1, 0, 0]) / np.sqrt(2),
            np.array([1, 0, 0, 1]) / np.sqrt(2),
            np.array([1j, 1j, 1j, 1j]) * 0.5,
            np.array([0, 1]),
            np.array([1, 0, 1, 0, 0, 1, 1, 0]) / 2,
        ],
    )
    def test_MultiPhaseOperation_with_all_parameters_set_to_0_behaves_like_identity(
        self, wavefunction
    ):
        params = tuple(np.zeros_like(wavefunction))
        operation = MultiPhaseOperation(params)

        np.testing.assert_array_equal(operation.apply(wavefunction), wavefunction)

    @pytest.mark.parametrize(
        "wavefunction,params",
        [
            [np.array([1, 0, 0, 0]), np.array([])],
            [np.array([1, 1, 0, 0]) / np.sqrt(2), np.array([0, 1, 2])],
            [np.array([1, 0, 0, 1]) / np.sqrt(2), np.array([0, 1, 2, 3, 4])],
        ],
    )
    def test_MultiPhaseOperation_apply_fails_if_params_and_wavefunction_dont_match(
        self, wavefunction, params
    ):
        operation = MultiPhaseOperation(params)
        with pytest.raises(ValueError):
            operation.apply(wavefunction)

    @pytest.mark.parametrize(
        "wavefunction,params",
        [
            [np.array([1, 1, 1, 1]) / 2, np.array([3j, 1, 2, 0])],
            [np.array([1, 1, 1, 1]) / 2, np.array([0, 1 + 1j, 2, 3j])],
        ],
    )
    def test_MultiPhaseOperation_apply_fails_if_params_are_not_real(
        self, wavefunction, params
    ):
        pytest.xfail("Not sure why")
        operation = MultiPhaseOperation(params)
        with pytest.raises(RuntimeError):
            _ = operation.apply(wavefunction)

    @pytest.mark.parametrize(
        "wavefunction,params,target_wavefunction",
        [
            [
                np.array([1, 1, 1, 1]) / 2,
                np.array([np.pi, np.pi / 2, np.pi / 3, np.pi / 4]),
                np.array(
                    [
                        np.exp(np.pi * 1j),
                        np.exp(np.pi / 2 * 1j),
                        np.exp(np.pi / 3 * 1j),
                        np.exp(np.pi / 4 * 1j),
                    ]
                )
                / 2,
            ],
            [
                np.array([1, 0, 0, 1]) / np.sqrt(2),
                np.array([np.pi, np.pi / 2, np.pi / 3, np.pi / 4]),
                np.array(
                    [
                        np.exp(np.pi * 1j),
                        0,
                        0,
                        np.exp(np.pi / 4 * 1j),
                    ]
                )
                / np.sqrt(2),
            ],
            [
                np.array([1, 1, 1, 1]) / 2,
                np.array([np.pi, 0, 0, 0]),
                np.array(
                    [
                        np.exp(np.pi * 1j),
                        1,
                        1,
                        1,
                    ]
                )
                / 2,
            ],
        ],
    )
    def test_MultiPhaseOperation_apply_works(
        self, wavefunction, params, target_wavefunction
    ):
        operation = MultiPhaseOperation(params)
        output = operation.apply(wavefunction)
        np.testing.assert_allclose(output, target_wavefunction)

    @pytest.mark.parametrize(
        "wavefunction,params,target_wavefunction",
        [
            [
                np.array([1, 1, 1, 1]) / 2,
                np.array(
                    [
                        Symbol("alpha_0"),
                        Symbol("alpha_1"),
                        Symbol("alpha_2"),
                        Symbol("alpha_3"),
                    ]
                ),
                np.array(
                    [
                        np.exp(Symbol("alpha_0") * 1j),
                        np.exp(Symbol("alpha_1") * 1j),
                        np.exp(Symbol("alpha_2") * 1j),
                        np.exp(Symbol("alpha_3") * 1j),
                    ]
                )
                / 2,
            ],
        ],
    )
    def test_MultiPhaseOperation_apply_works_with_symbolic_params(
        self, wavefunction, params, target_wavefunction
    ):
        operation = MultiPhaseOperation(params)
        output = operation.apply(wavefunction)
        # TODO
        # np.testing.assert_allclose(output, target_wavefunction)
