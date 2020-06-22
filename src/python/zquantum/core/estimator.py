from .interfaces.estimator import Estimator
from .interfaces.backend import QuantumBackend, QuantumSimulator
from .circuit import Circuit
from .measurement import ExpectationValues
from openfermion import SymbolicOperator


class BasicEstimator(Estimator):
    """An estimator that uses the standard approach to computing expectation values of an operator.
    """

    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: int = None,
        epsilon: float = None,
        delta: float = None,
    ) -> ExpectationValues:
        """Given a circuit, backend, and target operators, this method produces expectation values 
        for each target operator using the get_expectation_values method built into the provided QuantumBackend. 

        Args:
            backend (QuantumBackend): the backend that will be used to run the circuit
            circuit (Circuit): the circuit that prepares the state.
            target_operator (List[SymbolicOperator]): List of target functions to be estimated.
            n_samples (int): Number of measurements done. 
            epsilon (float): an error term.
            delta (float): a confidence term.

        Returns:
            ExpectationValues: expectation values for each term in the target operator.
        """
        estimator_name = type(self).__name__
        self._log_ignore_parameter(estimator_name, "epsilon", epsilon)
        self._log_ignore_parameter(estimator_name, "delta", delta)

        if n_samples is not None:
            self.logger.warning(
                "Using n_samples={} (arugment passed to get_estimated_expectation_values). Ignoring backend.n_samples={}.".format(
                    n_samples, backend.n_samples
                )
            )
            saved_n_samples = backend.n_samples
            backend.n_samples = n_samples
            expectation_values = backend.get_expectation_values(
                circuit, target_operator
            )
            backend.n_samples = saved_n_samples
            return expectation_values
        return backend.get_expectation_values(circuit, target_operator)


class ExactEstimator(Estimator):
    """An estimator that exactly computes the expectation values of an operator. This estimator must run on a quantum simulator. 
    """

    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: int = None,
        epsilon: float = None,
        delta: float = None,
    ) -> ExpectationValues:
        """Given a circuit, backend, and target operators, this method produces expectation values 
        for each target operator using the get_exact_expectation_values method built into the provided QuantumBackend. 

        Args:
            backend (QuantumBackend): the backend that will be used to run the circuit
            circuit (Circuit): the circuit that prepares the state.
            target_operator (List[SymbolicOperator]): List of target functions to be estimated.
            n_samples (int): Number of measurements done on the unknown quantum state. 
            epsilon (float): an error term.
            delta (float): a confidence term.

        Raises:
            AttributeError: If backend is not a QuantumSimulator. 

        Returns:
            ExpectationValues: expectation values for each term in the target operator.
        """
        estimator_name = type(self).__name__
        self._log_ignore_parameter(estimator_name, "n_samples", n_samples)
        self._log_ignore_parameter(estimator_name, "epsilon", epsilon)
        self._log_ignore_parameter(estimator_name, "delta", delta)

        if isinstance(backend, QuantumSimulator):
            return backend.get_exact_expectation_values(circuit, target_operator)
        else:
            raise AttributeError(
                "To use the ExactEstimator, the backend must be a QuantumSimulator."
            )
