from zquantum.core.wip.interfaces.ansatz import Ansatz
from zquantum.core.wip import circuits

import pytest
import sympy


class TestImplementingViaInheritance:
    def test_empty_class_raises_type_error(self):
        class ExampleAnsatz(Ansatz):
            pass

        with pytest.raises(TypeError):
            _ = ExampleAnsatz()

    def test_impementing_all_methods_doesnt_raise(self):
        class ExampleAnsatz(Ansatz):
            number_of_layers = 1
            supports_parametrized_circuits = True
            number_of_qubits = 3
            number_of_params = 2
            parametrized_circuit = circuits.Circuit(
                [
                    circuits.RX(sympy.Symbol("theta"))(0),
                    circuits.RX(sympy.Symbol("gamma"))(2),
                ]
            )

            def generate_circuit(self, params) -> circuits.Circuit:
                return self.parametrized_circuit

            def get_executable_circuit(self, params) -> circuits.Circuit:
                return self.parametrized_circuit.bind(
                    {sympy.Symbol("gamma"): params[0], sympy.Symbol("theta"): params[1]}
                )

        _ = ExampleAnsatz()

    def test_not_impementing_property_raises_type_error(self):
        class ExampleAnsatz(Ansatz):
            number_of_layers = 1
            # supports_parametrized_circuits = True
            number_of_qubits = 3
            number_of_params = 2
            parametrized_circuit = circuits.Circuit(
                [
                    circuits.RX(sympy.Symbol("theta"))(0),
                    circuits.RX(sympy.Symbol("gamma"))(2),
                ]
            )

            def generate_circuit(self, params) -> circuits.Circuit:
                return self.parametrized_circuit

            def get_executable_circuit(self, params) -> circuits.Circuit:
                return self.parametrized_circuit.bind(
                    {sympy.Symbol("gamma"): params[0], sympy.Symbol("theta"): params[1]}
                )

        with pytest.raises(TypeError):
            _ = ExampleAnsatz()


# def test_implementing_free_form():
#     class ExampleAnsatz:
#         pass

#     example: Ansatz = ExampleAnsatz()

#     example: Ansatz = ExampleAnsatz()
