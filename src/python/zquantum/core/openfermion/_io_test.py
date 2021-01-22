import unittest
import subprocess
import os
import numpy as np

from openfermion import (
    QubitOperator,
    FermionOperator,
    IsingOperator,
    get_interaction_operator,
    hermitian_conjugated,
    InteractionRDM,
)
from ..circuit import build_uniform_param_grid, save_circuit_template_params
from ..interfaces.mock_objects import MockAnsatz
from ..utils import SCHEMA_VERSION, convert_dict_to_array, create_object

from ._utils import evaluate_operator_for_parameter_grid
from ._io import (
    load_qubit_operator,
    save_qubit_operator,
    load_qubit_operator_set,
    save_qubit_operator_set,
    load_interaction_operator,
    save_interaction_operator,
    convert_qubitop_to_dict,
    get_pauli_strings,
    convert_dict_to_qubitop,
    convert_interaction_op_to_dict,
    convert_dict_to_interaction_op,
    convert_isingop_to_dict,
    convert_dict_to_isingop,
    save_ising_operator,
    load_ising_operator,
    save_parameter_grid_evaluation,
    save_interaction_rdm,
    load_interaction_rdm,
    convert_interaction_rdm_to_dict,
    convert_dict_to_interaction_rdm,
)


class TestQubitOperator(unittest.TestCase):
    def setUp(self):
        n_modes = 2
        np.random.seed(0)
        one_body_tensor = np.random.rand(*(n_modes,) * 2)
        two_body_tensor = np.random.rand(*(n_modes,) * 4)
        self.interaction_rdm = InteractionRDM(one_body_tensor, two_body_tensor)

    def test_qubitop_to_dict_io(self):
        # Given
        qubit_op = QubitOperator(((0, "Y"), (1, "X"), (2, "Z"), (4, "X")), 3.0j)
        qubit_op += hermitian_conjugated(qubit_op)

        # When
        qubitop_dict = convert_qubitop_to_dict(qubit_op)
        recreated_qubit_op = convert_dict_to_qubitop(qubitop_dict)

        # Then
        self.assertEqual(recreated_qubit_op, qubit_op)

    def test_qubit_operator_io(self):
        # Given
        qubit_op = QubitOperator(((0, "Y"), (3, "X"), (8, "Z"), (11, "X")), 3.0j)

        # When
        save_qubit_operator(qubit_op, "qubit_op.json")
        loaded_op = load_qubit_operator("qubit_op.json")

        # Then
        self.assertEqual(qubit_op, loaded_op)

    def test_qubit_operator_set_io(self):
        qubit_op1 = QubitOperator(((0, "Y"), (3, "X"), (8, "Z"), (11, "X")), 3.0j)
        qubit_op2 = QubitOperator(((0, "Y"), (0, "X"), (7, "Z"), (14, "X")), 1.0j)

        qubit_operator_set = [qubit_op1, qubit_op2]
        save_qubit_operator_set(qubit_operator_set, "qubit_operator_set.json")
        loaded_qubit_operator_set = load_qubit_operator_set("qubit_operator_set.json")
        for i in range(len(qubit_operator_set)):
            self.assertEqual(qubit_operator_set[i], loaded_qubit_operator_set[i])
        os.remove("qubit_operator_set.json")

    def test_interaction_op_to_dict_io(self):
        # Given
        test_op = FermionOperator("1^ 2^ 3 4")
        test_op += hermitian_conjugated(test_op)
        interaction_op = get_interaction_operator(test_op)
        interaction_op.constant = 0.0

        # When
        interaction_op_dict = convert_interaction_op_to_dict(interaction_op)
        recreated_interaction_op = convert_dict_to_interaction_op(interaction_op_dict)

        # Then
        self.assertEqual(recreated_interaction_op, interaction_op)

    def test_interaction_operator_io(self):
        # Given
        test_op = FermionOperator("1^ 2^ 3 4")
        test_op += hermitian_conjugated(test_op)
        interaction_op = get_interaction_operator(test_op)
        interaction_op.constant = 0.0

        # When
        save_interaction_operator(interaction_op, "interaction_op.json")
        loaded_op = load_interaction_operator("interaction_op.json")

        # Then
        self.assertEqual(interaction_op, loaded_op)
        os.remove("interaction_op.json")

    def test_qubitop_io(self):
        # Given
        qubit_op = QubitOperator(((0, "Y"), (1, "X"), (2, "Z"), (4, "X")), 3.0j)

        # When
        save_qubit_operator(qubit_op, "qubit_op.json")
        loaded_op = load_qubit_operator("qubit_op.json")

        # Then
        self.assertEqual(qubit_op, loaded_op)
        os.remove("qubit_op.json")

    def test_get_pauli_strings(self):
        qubit_operator = (
            QubitOperator(((0, "X"), (1, "Y")))
            - 0.5 * QubitOperator(((1, "Y"),))
            + 0.5 * QubitOperator(())
        )
        constructed_list = get_pauli_strings(qubit_operator)
        target_list = ["X0Y1", "Y1", ""]
        self.assertListEqual(constructed_list, target_list)

    def test_isingop_to_dict_io(self):
        # Given
        ising_op = IsingOperator("[] + 3[Z0 Z1] + [Z1 Z2]")

        # When
        isingop_dict = convert_isingop_to_dict(ising_op)
        recreated_isingop = convert_dict_to_isingop(isingop_dict)

        # Then
        self.assertEqual(recreated_isingop, ising_op)

    def test_isingop_io(self):
        # Given
        ising_op = IsingOperator("[] + 3[Z0 Z1] + [Z1 Z2]")

        # When
        save_ising_operator(ising_op, "ising_op.json")
        loaded_op = load_ising_operator("ising_op.json")

        # Then
        self.assertEqual(ising_op, loaded_op)
        os.remove("ising_op.json")

    def test_save_parameter_grid_evaluation(self):
        # Given
        ansatz = MockAnsatz(2, 2)
        grid = build_uniform_param_grid(1, 2, 0, np.pi, np.pi / 10)
        backend = create_object(
            {
                "module_name": "zquantum.core.interfaces.mock_objects",
                "function_name": "MockQuantumSimulator",
            }
        )
        op = QubitOperator("0.5 [] + 0.5 [Z1]")
        (
            parameter_grid_evaluation,
            optimal_parameters,
        ) = evaluate_operator_for_parameter_grid(ansatz, grid, backend, op)
        # When
        save_parameter_grid_evaluation(
            parameter_grid_evaluation, "parameter-grid-evaluation.json"
        )
        save_circuit_template_params(optimal_parameters, "optimal-parameters.json")
        # Then
        # TODO

    def test_interaction_rdm_io(self):
        # Given

        # When
        save_interaction_rdm(self.interaction_rdm, "interaction_rdm.json")
        loaded_interaction_rdm = load_interaction_rdm("interaction_rdm.json")

        # Then
        self.assertEqual(self.interaction_rdm, loaded_interaction_rdm)
        os.remove("interaction_rdm.json")

    def test_convert_interaction_rdm_to_dict(self):
        rdm_dict = convert_interaction_rdm_to_dict(self.interaction_rdm)

        self.assertEqual(rdm_dict["schema"], SCHEMA_VERSION + "-interaction_rdm")
        self.assertTrue(
            np.allclose(
                convert_dict_to_array(rdm_dict["one_body_tensor"]),
                self.interaction_rdm.one_body_tensor,
            )
        )
        self.assertTrue(
            np.allclose(
                convert_dict_to_array(rdm_dict["two_body_tensor"]),
                self.interaction_rdm.two_body_tensor,
            )
        )

    def test_convert_dict_to_interaction_rdm(self):
        rdm_dict = convert_interaction_rdm_to_dict(self.interaction_rdm)
        converted_interaction_rdm = convert_dict_to_interaction_rdm(rdm_dict)

        self.assertEqual(self.interaction_rdm, converted_interaction_rdm)

    def tearDown(self):
        subprocess.run(
            ["rm", "parameter-grid-evaluation.json", "optimal-parameters.json"]
        )
