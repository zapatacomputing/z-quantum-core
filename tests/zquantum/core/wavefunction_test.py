from math import sqrt

import numpy as np
import pytest
from sympy import I, Matrix, Symbol, cos, exp, sin
from zquantum.core.circuits._builtin_gates import RX, RY, U3, H, X
from zquantum.core.circuits._circuit import Circuit
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.core.testing import create_random_wavefunction
from zquantum.core.wavefunction import Wavefunction


class TestInitializations:
    def test_init_system_returns_numpy_array(self):
        wf = Wavefunction.init_system(2)
        assert isinstance(wf._amplitude_vector, np.ndarray)

    def test_constructor_returns_numpy_array_for_no_symbols(self):
        wf = Wavefunction([1.0, 0, 0, 0])
        assert isinstance(wf._amplitude_vector, np.ndarray)

    def test_constructor_returns_sympy_matrix_for_free_symbols(self):
        wf = Wavefunction([0.25, 0, Symbol("alpha"), 0])
        assert isinstance(wf._amplitude_vector, Matrix)

    @pytest.mark.parametrize(
        "input_list", [[], np.zeros(17), create_random_wavefunction(3)[:-1]]
    )
    def test_init_fails_when_len_of_passed_list_is_not_power_of_two(self, input_list):
        with pytest.raises(ValueError):
            Wavefunction(input_list)

    @pytest.mark.parametrize("input_list", [np.ones(8), np.zeros(4)])
    def test_init_fails_when_passed_list_has_no_free_symbols_and_no_unity(
        self, input_list
    ):
        with pytest.raises(ValueError):
            Wavefunction(input_list)

    @pytest.mark.parametrize(
        "input_list",
        [
            [Symbol("alpha"), 2.0],
            [
                Symbol("alpha"),
                Symbol("beta"),
                sqrt(3) / 2,
                sqrt(3) / 2,
            ],
        ],
    )
    def test_init_fails_when_passed_list_has_free_symbols_and_exceeds_unity(
        self, input_list
    ):
        with pytest.raises(ValueError):
            Wavefunction(input_list)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
    def test_init_system_returns_expected_wavefunction_size(self, n_qubits):
        wavefunction = Wavefunction.init_system(n_qubits=n_qubits)

        # Check length
        assert len(wavefunction) == 2 ** n_qubits

        # Check internal property
        assert wavefunction.n_qubits == n_qubits

        # Check amplitude of zero state
        assert wavefunction[0] == 1.0

        # Check amplitude of the rest of the states
        assert not np.any(wavefunction[1:])

    def test_init_system_raises_warning_for_non_ints(self):
        with pytest.warns(UserWarning):
            Wavefunction.init_system(1.234)

    @pytest.mark.parametrize("n_qubits", [0, -1, -2])
    def test_init_system_fails_on_invalid_params(self, n_qubits):
        with pytest.raises(ValueError):
            Wavefunction.init_system(n_qubits=n_qubits)


class TestFunctions:
    @pytest.fixture
    def symbolic_wf(self) -> Wavefunction:
        return Wavefunction([Symbol("alpha"), 0.5, Symbol("beta"), 0.5])

    @pytest.fixture
    def numeric_wf(self) -> Wavefunction:
        return Wavefunction([0.5, 0.5, 0.5, 0.5])

    @pytest.mark.parametrize("new_val", [1.0, -1.0])
    def test_set_item_raises_error_for_invalid_sets(self, symbolic_wf, new_val):
        with pytest.raises(ValueError):
            symbolic_wf[0] = new_val

    @pytest.mark.parametrize("new_value", [0.5, Symbol("gamma")])
    def test_set_item_passes_if_still_below_unity(self, symbolic_wf, new_value):
        symbolic_wf[0] = new_value

        assert symbolic_wf[0] == new_value

    def test_iterator(self, symbolic_wf):
        for i, elem in enumerate(symbolic_wf):
            assert elem == symbolic_wf[i]

    @pytest.mark.parametrize(
        "symbol_map", [{"alpha": 1.0}, {"alpha": 0.5, "beta": 0.6}]
    )
    def test_bindings_fail_like_setitem(self, symbolic_wf, symbol_map):
        with pytest.raises(ValueError):
            symbolic_wf.bind(symbol_map)

    def test_bind_returns_new_object_for_symbolic_wf(self, symbolic_wf):
        assert symbolic_wf is not symbolic_wf.bind({})

    def test_bind_does_not_return_new_object_for_numeric_wf(self, numeric_wf):
        assert numeric_wf is numeric_wf.bind({})

    def test_binding_all_symbols_returns_numpy_array(self, symbolic_wf: Wavefunction):
        assert isinstance(
            symbolic_wf.bind({"alpha": 0.5, "beta": 0.5})._amplitude_vector, np.ndarray
        )

    @pytest.mark.parametrize("other_obj", [[], np.zeros(8)])
    def test_eq_returns_false_for_non_wavefunction_objects(
        self, symbolic_wf, numeric_wf, other_obj
    ):
        assert not (symbolic_wf == other_obj)
        assert not (numeric_wf == other_obj)

    def test_eq_returns_true_for_objects_with_equal_wavefunctions(
        self, symbolic_wf: Wavefunction, numeric_wf: Wavefunction
    ):
        test_wf = Wavefunction(symbolic_wf._amplitude_vector)
        assert symbolic_wf == test_wf

        test_wf = Wavefunction(numeric_wf._amplitude_vector)
        np.testing.assert_array_equal(numeric_wf, test_wf)


class TestRepresentations:
    def test_string_output_of_symbolic_wavefunction(self):
        wf = Wavefunction([Symbol("alpha"), 0])

        wf_str = wf.__str__()

        assert "alpha" in wf_str
        assert wf_str.endswith("])")
        assert wf_str.startswith("Wavefunction([")

    def test_string_output_of_numeric_wavefunction(self):
        wf = Wavefunction([1j, 0])

        wf_str = wf.__str__()

        assert "j" in wf_str
        assert wf_str.endswith("])")
        assert wf_str.startswith("Wavefunction([")

    @pytest.mark.parametrize(
        "wf", [Wavefunction.init_system(2), Wavefunction([Symbol("alpha"), 0.0])]
    )
    def test_amplitudes_and_probs_output_type(self, wf: Wavefunction):
        if len(wf.free_symbols) > 0:
            assert wf.amplitudes.dtype == object
            assert wf.probabilities().dtype == object
        else:
            assert wf.amplitudes.dtype == np.complex128
            assert wf.probabilities().dtype == np.float64

    @pytest.mark.parametrize(
        "wf_vec",
        [
            [1.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [1 / sqrt(2), 0, 0, 0, 0, 0, 0, 1 / sqrt(2)],
        ],
    )
    def test_get_outcome_probs(self, wf_vec):
        wf = Wavefunction(wf_vec)
        probs_dict = wf.get_outcome_probs()

        assert all([len(key) == wf.n_qubits for key in probs_dict.keys()])

        for key in probs_dict.keys():
            assert len(key) == wf.n_qubits

            assert wf.probabilities()[int(key, 2)] == probs_dict[key]


class TestGates:
    @pytest.fixture
    def simulator(self) -> SymbolicSimulator:
        return SymbolicSimulator()

    @pytest.mark.parametrize(
        "circuit, expected_wavefunction",
        [
            (
                Circuit([RX(Symbol("theta"))(0)]),
                Wavefunction(
                    [1.0 * cos(Symbol("theta") / 2), -1j * sin(Symbol("theta") / 2)]
                ),
            ),
            (
                Circuit([X(0), RY(Symbol("theta"))(0)]),
                Wavefunction(
                    [
                        -1.0 * sin(Symbol("theta") / 2),
                        1.0 * cos(Symbol("theta") / 2),
                    ]
                ),
            ),
            (
                Circuit(
                    [H(0), U3(Symbol("theta"), Symbol("phi"), Symbol("lambda"))(0)]
                ),
                Wavefunction(
                    [
                        cos(Symbol("theta") / 2) / sqrt(2)
                        + -exp(I * Symbol("lambda"))
                        * sin(Symbol("theta") / 2)
                        / sqrt(2),
                        exp(I * Symbol("phi")) * sin(Symbol("theta") / 2) / sqrt(2)
                        + exp(I * (Symbol("lambda") + Symbol("phi")))
                        * cos(Symbol("theta") / 2)
                        / sqrt(2),
                    ]
                ),
            ),
        ],
    )
    def test_wavefunction_works_as_expected_with_symbolic_circuits(
        self,
        simulator: SymbolicSimulator,
        circuit: Circuit,
        expected_wavefunction: Wavefunction,
    ):
        returned_wavefunction = simulator.get_wavefunction(circuit)

        assert returned_wavefunction == expected_wavefunction
