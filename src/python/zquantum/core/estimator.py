from .interfaces.estimator import Estimator
from .interfaces.backend import QuantumBackend, QuantumSimulator
from .circuit import Circuit
from .measurement import (
    ExpectationValues,
    expectation_values_to_real,
    concatenate_expectation_values,
)
from .hamiltonian import group_comeasureable_terms_greedy, compute_group_variances
from openfermion import SymbolicOperator, IsingOperator, QubitOperator
from overrides import overrides
import logging
import numpy as np
import pyquil
from typing import Tuple, Optional, Callable, List

logger = logging.getLogger(__name__)

DECOMPOSITION_METHODS = {
    "greedy": group_comeasureable_terms_greedy,
    "greedy-sorted": lambda qubit_operator: group_comeasureable_terms_greedy(
        qubit_operator, True
    ),
}


def get_decomposition_function(
    decomposition_method: str,
) -> Callable[[QubitOperator], List[QubitOperator]]:
    """Get a function for Hamiltonian decomposition from its name.

    Args:
        decomposition_method: The name of the Hamiltonian decomposition method.
    
    Returns:
        A callable that performs the decomposition.
    """

    decomposition_function = DECOMPOSITION_METHODS.get(decomposition_method)
    if decomposition_function is None:
        raise ValueError(
            f"Unrecognized decomposition method {decomposition_method}. Allowed values are {list(DECOMPOSITION_METHODS.keys())}"
        )
    return decomposition_function


def get_context_selection_circuit(
    term: Tuple[Tuple[int, str], ...]
) -> Tuple[Circuit, IsingOperator]:
    """Get the context selection circuit for measuring the expectation value
    of a Pauli term.

    Args:
        term: The Pauli term, expressed using the OpenFermion convention.
    
    Returns:
        Tuple containing:
        - The context selection circuit.
        - The frame operator
    """

    context_selection_circuit = Circuit()
    operator = IsingOperator(())
    for factor in term:
        if factor[1] == "X":
            context_selection_circuit += Circuit(pyquil.gates.RY(-np.pi / 2, factor[0]))
        elif factor[1] == "Y":
            context_selection_circuit += Circuit(pyquil.gates.RX(np.pi / 2, factor[0]))
        operator *= IsingOperator((factor[0], "Z"))

    return context_selection_circuit, operator


def get_context_selection_circuit_for_group(
    qubit_operator: QubitOperator,
) -> Tuple[Circuit, IsingOperator]:
    """Get the context selection circuit for measuring the expectation value
    of a group of co-measurable Pauli terms.

    Args:
        term: The Pauli term, expressed using the OpenFermion convention.
    
    Returns:
        Tuple containing:
        - The context selection circuit.
        - The frame operator
    """

    context_selection_circuit = Circuit()
    transformed_operator = IsingOperator()
    context = []

    for term in qubit_operator.terms:
        term_operator = IsingOperator(())
        for qubit, operator in term:
            for existing_qubit, existing_operator in context:
                if existing_qubit == qubit and existing_operator != operator:
                    raise ValueError("Terms are not co-measurable")
            if not (qubit, operator) in context:
                context.append((qubit, operator))
            term_operator *= IsingOperator((qubit, "Z"))
        transformed_operator += term_operator * qubit_operator.terms[term]

    for factor in context:
        if factor[1] == "X":
            context_selection_circuit += Circuit(pyquil.gates.RY(-np.pi / 2, factor[0]))
        elif factor[1] == "Y":
            context_selection_circuit += Circuit(pyquil.gates.RX(np.pi / 2, factor[0]))

    return context_selection_circuit, transformed_operator

def estimate_nmeas(
    target_operator: QubitOperator,
    decomposition_method: Optional[str] = "greedy-sorted",
    expecval: Optional[ExpectationValues] = None,
) -> Tuple[float, int, np.array]:
    """Calculates the number of measurements required for computing
        the expectation value of a qubit hamiltonian, where co-measurable terms
        are grouped. We're assuming the exact expectation values are provided
        (i.e. infinite number of measurements or simulations without noise)
        M ~ (\sum_{i} prec(H_i)) ** 2.0 / (epsilon ** 2.0)
        where prec(H_i) is the precision (square root of the variance)
        for each group of co-measurable terms H_{i}. It is computed as
        prec(H_{i}) = \sum{ab} |h_{a}^{i}||h_{b}^{i}| cov(O_{a}^{i}, O_{b}^{i})
        where h_{a}^{i} is the coefficient of the a-th operator, O_{a}^{i}, in the
        i-th group. Covariances are assumed to be zero for a != b:
        cov(O_{a}^{i}, O_{b}^{i}) = <O_{a}^{i} O_{b}^{i}> - <O_{a}^{i}> <O_{b}^{i}> = 0
    Args:
        target_operator (openfermion.ops.QubitOperator): A QubitOperator to measure
        expecval (ExpectationValues): An ExpectationValues object containing the expectation
                  values of the operators and their squares. Optionally, contains
                  values of operator products to compute covariances.
                  If absent, covariances are assumed to be 0 and variances are
                  assumed to be maximal, i.e. 1.
                  NOTE: IN THE CURRENT IMPLEMENTATION WE HAVE TO MAKE SURE THAT THE ORDER
                  OF EXPECTATION VALUES IS CONSISTENT WITH THE ORDER OF THE TERMS IN THE
                  TARGET QUBIT OPERATOR, OTHERWISE THIS FUNCTION WILL NOT WORK CORRECTLY
    Returns:
        K2 (float): number of measurements for epsilon = 1.0
        nterms (int): number of groups of QWC terms in the target_operator
        frame_meas (array): Number of optimal measurements per group 
    """

    frame_variances = None
    groups = get_decomposition_function(self.decomposition_method)(target_operator)
    frame_variances = compute_group_variances(groups, expecval)
    # Here we have our current best estimate for frame variances.
    # We first compute the measurement estimate for each frame

    sqrt_lambda = sum(np.sqrt(frame_variances))
    frame_meas = np.asarray([sqrt_lambda * np.sqrt(x) for x in frame_variances])
    K2 = sum(frame_meas)
    nterms = sum([len(group.terms) for group in groups])

    return K2, nterms, frame_meas

class BasicEstimator(Estimator):
    """An estimator that uses the standard approach to computing expectation values of an operator.
    
        Attributes:
            decomposition_method (str): Which Hamiltonian decomposition method
                to use. Available options are: 'greedy-sorted' (default) and
                'greedy'.
    """

    def __init__(self, decomposition_method: str = "greedy-sorted"):
        self.decomposition_method = decomposition_method

    @overrides
    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: Optional[int] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
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
        frame_operators = []
        frame_circuits = []
        groups = get_decomposition_function(self.decomposition_method)(target_operator)
        for group in groups:
            frame_circuit, frame_operator = get_context_selection_circuit_for_group(
                group
            )
            frame_circuits.append(circuit + frame_circuit)
            frame_operators.append(frame_operator)

        if n_samples is not None:
            logger.warning(
                f"""Using n_samples={n_samples} (argument passed to get_estimated_expectation_values). 
                    Ignoring backend.n_samples={backend.n_samples}"""
            )
            saved_n_samples = backend.n_samples
            backend.n_samples = n_samples
            measurements_set = backend.run_circuitset_and_measure(frame_circuits)
            backend.n_samples = saved_n_samples
        else:
            measurements_set = backend.run_circuitset_and_measure(frame_circuits)

        expectation_values_set = []
        for frame_operator, measurements in zip(frame_operators, measurements_set):
            expectation_values_set.append(
                expectation_values_to_real(
                    measurements.get_expectation_values(frame_operator)
                )
            )

        return expectation_values_to_real(
            concatenate_expectation_values(expectation_values_set)
        )


class ExactEstimator(Estimator):
    """An estimator that exactly computes the expectation values of an operator. This estimator must run on a quantum simulator. 
    """

    @overrides
    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: Optional[int] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
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
        if isinstance(backend, QuantumSimulator):
            return backend.get_exact_expectation_values(circuit, target_operator)
        else:
            raise AttributeError(
                "To use the ExactEstimator, the backend must be a QuantumSimulator."
            )
