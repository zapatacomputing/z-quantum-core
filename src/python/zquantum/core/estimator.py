from .interfaces.estimator import Estimator
from .interfaces.backend import QuantumBackend
from .circuit import Circuit
from .measurement import ExpectationValues
from openfermion import SymbolicOperator


class BasicEstimator(Estimator):
    """An estimator that uses the standard approach to computing expectation values of an operator.
    """

    def get_estimated_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
    ) -> ExpectationValues:
        """Given a circuit, backend, and target operators, this method produces expectation values 
        for each target operator using the get_expectation_values method built into the provided QuantumBackend. 

        Args:
            backend (QuantumBackend): the backend that will be used to run the circuit
            circuit (Circuit): the circuit that prepares the state.
            target_operator (List[SymbolicOperator]): List of target functions to be estimated.

        Returns:
            ExpectationValues: expectation values for each term in the target operator.
        """
        return backend.get_expectation_values(circuit, target_operator)
