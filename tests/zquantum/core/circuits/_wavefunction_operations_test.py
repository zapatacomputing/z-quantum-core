import numpy as np
import pytest
import sympy
from zquantum.core.circuits import MultiPhaseOperation


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
    def test_with_all_parameters_set_to_0_behaves_like_identity(self, wavefunction):
        params = tuple(np.zeros_like(wavefunction))
        operation = MultiPhaseOperation(params)

        np.testing.assert_array_equal(operation.apply(wavefunction), wavefunction)

    @pytest.mark.parametrize(
        "wavefunction,params",
        [
            (np.array([1, 0, 0, 0]), ()),
            (np.array([1, 1, 0, 0]) / np.sqrt(2), (0, 1, 2)),
            (np.array([1, 0, 0, 1]) / np.sqrt(2), (0, 1, 2, 3, 4)),
        ],
    )
    def test_cannot_be_applied_if_length_of_params_and_wavefunction_dont_match(
        self, wavefunction, params
    ):
        operation = MultiPhaseOperation(params)
        with pytest.raises(ValueError):
            operation.apply(wavefunction)

    @pytest.mark.parametrize(
        "wavefunction,params",
        [
            (np.array([1, 1, 1, 1]) / 2, (3j, 1, 2, 0)),
            (np.array([1, 1, 1, 1]) / 2, (0, 1 + 1j, 2, 3j)),
        ],
    )
    def test_cannot_be_applied_if_params_are_not_real(self, wavefunction, params):
        with pytest.raises(ValueError):
            MultiPhaseOperation(params)

    @pytest.mark.parametrize(
        "wavefunction,params,target_wavefunction",
        [
            [
                np.array([1, 1, 1, 1]) / 2,
                np.array([np.pi, -np.pi / 2, np.pi / 3, -np.pi / 4]),
                np.array(
                    [
                        np.exp(np.pi * 1j),
                        np.exp(-np.pi / 2 * 1j),
                        np.exp(np.pi / 3 * 1j),
                        np.exp(-np.pi / 4 * 1j),
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
            [
                np.array([1, 1, 1, 1]) / 2,
                np.array([sympy.pi, sympy.sympify(0), 0, 0]),
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
    def test_correctly_transforms_wavefunction(
        self, wavefunction, params, target_wavefunction
    ):
        operation = MultiPhaseOperation(params)
        output = operation.apply(wavefunction)
        np.testing.assert_allclose(output, target_wavefunction)

    @pytest.mark.parametrize(
        "wavefunction",
        [np.array([1, 1, 1, 1]) / 2, np.array([1, 0, 0, 1]) / np.sqrt(2)],
    )
    @pytest.mark.parametrize(
        "params", [tuple(sympy.symbols("alpha_0:4")), (sympy.Symbol("theta"), 1, 1, 1)]
    )
    def test_apply_does_not_work_with_symbolic_params(self, wavefunction, params):
        operation = MultiPhaseOperation(params)
        with pytest.raises(RuntimeError):
            operation.apply(wavefunction)

    @pytest.mark.parametrize(
        "params, expected_free_symbols",
        [
            ((0, 1, 1, -2), []),
            (
                (sympy.Symbol("a"), sympy.Symbol("b")),
                [sympy.Symbol("a"), sympy.Symbol("b")],
            ),
            (
                (
                    1,
                    sympy.Symbol("a") + sympy.Symbol("d"),
                    -1,
                    sympy.Symbol("c") - sympy.Symbol("b"),
                ),
                list(sympy.symbols("a, b, c, d")),
            ),
        ],
    )
    def test_free_symbols_in_parameters_are_correctly_reported(
        self, params, expected_free_symbols
    ):
        operation = MultiPhaseOperation(params)
        assert operation.free_symbols == expected_free_symbols
