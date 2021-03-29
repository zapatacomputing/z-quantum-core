import json
from zquantum.core.utils import (
    create_object,
    load_noise_model,
    save_value_estimate,
    save_nmeas_estimate,
    save_list,
)
from zquantum.core.measurement import load_expectation_values, save_expectation_values
from zquantum.core.hamiltonian import (
    estimate_nmeas_for_operator,
    estimate_nmeas_for_frames,
    get_expectation_values_from_rdms,
    get_expectation_values_from_rdms_for_qubitoperator_list,
)
from zquantum.core.circuit import (
    load_circuit,
    load_circuit_connectivity,
    load_circuit_template_params,
    load_circuit_set,
    Circuit,
)
from zquantum.core.bitstring_distribution import save_bitstring_distribution
from zquantum.core.openfermion import (
    load_qubit_operator,
    load_interaction_rdm,
    load_qubit_operator_set,
)
from zquantum.core.typing import Specs
from typing import Dict, Optional, Union


def run_circuit_and_measure(
    backend_specs: Specs,
    circuit: Union[str, Dict],
    n_samples: Optional[int] = None,
    noise_model: Optional[str] = None,
    device_connectivity: Optional[str] = None,
):
    if isinstance(backend_specs, str):
        backend_specs = json.loads(backend_specs)
    if noise_model is not None:
        backend_specs["noise_model"] = load_noise_model(noise_model)
    if device_connectivity is not None:
        backend_specs["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )

    backend = create_object(backend_specs)
    if isinstance(circuit, str):
        circuit = load_circuit(circuit)
    else:
        circuit = Circuit.from_dict(circuit)

    measurements = backend.run_circuit_and_measure(circuit, n_samples=n_samples)
    measurements.save("measurements.json")


def run_circuitset_and_measure(
    backend_specs: Specs,
    circuitset: str,
    n_samples: Optional[int] = None,
    noise_model: Optional[str] = None,
    device_connectivity: Optional[str] = None,
):

    if isinstance(backend_specs, str):
        backend_specs = json.loads(backend_specs)
    if noise_model is not None:
        backend_specs["noise_model"] = load_noise_model(noise_model)
    if device_connectivity is not None:
        backend_specs["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )

    circuit_set = load_circuit_set(circuitset)
    backend = create_object(backend_specs)

    measurements_set = backend.run_circuitset_and_measure(
        circuit_set, n_samples=n_samples
    )
    list_of_measurements = [measurement.bitstrings for measurement in measurements_set]
    save_list(list_of_measurements, "measurements-set.json")


def get_bitstring_distribution(
    backend_specs: Specs,
    circuit: str,
    noise_model: Optional[str] = None,
    device_connectivity: Optional[str] = None,
):
    if isinstance(backend_specs, str):
        backend_specs = json.loads(backend_specs)
    if noise_model is not None:
        backend_specs["noise_model"] = load_noise_model(noise_model)
    if device_connectivity is not None:
        backend_specs["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )

    backend = create_object(backend_specs)
    circuit = load_circuit(circuit)

    bitstring_distribution = backend.get_bitstring_distribution(circuit)
    save_bitstring_distribution(bitstring_distribution, "bitstring-distribution.json")


def evaluate_ansatz_based_cost_function(
    ansatz_specs: Specs,
    backend_specs: Specs,
    cost_function_specs: Specs,
    ansatz_parameters: Specs,
    qubit_operator: str,
    noise_model: Optional[str] = None,
    device_connectivity: Optional[str] = None,
    prior_expectation_values: Optional[str] = None,
):
    ansatz_parameters = load_circuit_template_params(ansatz_parameters)
    # Load qubit op
    operator = load_qubit_operator(qubit_operator)
    if isinstance(ansatz_specs, str):
        ansatz_specs = json.loads(ansatz_specs)
    if ansatz_specs["function_name"] == "QAOAFarhiAnsatz":
        ansatz = create_object(ansatz_specs, cost_hamiltonian=operator)
    else:
        ansatz = create_object(ansatz_specs)

    if isinstance(backend_specs, str):
        backend_specs = json.loads(backend_specs)
    if noise_model is not None:
        backend_specs["noise_model"] = load_noise_model(noise_model)
    if device_connectivity is not None:
        backend_specs["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )

    backend = create_object(backend_specs)

    if isinstance(cost_function_specs, str):
        cost_function_specs = json.loads(cost_function_specs)
    estimator_specs = cost_function_specs.pop("estimator-specs", None)
    if estimator_specs is not None:
        cost_function_specs["estimator"] = create_object(estimator_specs)
    cost_function_specs["target_operator"] = operator
    cost_function_specs["ansatz"] = ansatz
    cost_function_specs["backend"] = backend
    cost_function = create_object(cost_function_specs)

    if prior_expectation_values is not None:
        if isinstance(prior_expectation_values, str):
            cost_function.estimator.prior_expectation_values = load_expectation_values(
                prior_expectation_values
            )

    value_estimate = cost_function(ansatz_parameters)

    save_value_estimate(value_estimate, "value_estimate.json")


def hamiltonian_analysis(
    qubit_operator: str,
    decomposition_method: str = "greedy",
    expectation_values: Optional[str] = None,
):
    operator = load_qubit_operator(qubit_operator)
    if expectation_values is not None:
        expecval = load_expectation_values(expectation_values)
    else:
        expecval = None

    K_coeff, nterms, frame_meas = estimate_nmeas_for_operator(
        operator, decomposition_method, expecval
    )
    save_nmeas_estimate(
        nmeas=K_coeff,
        nterms=nterms,
        frame_meas=frame_meas,
        filename="hamiltonian_analysis.json",
    )


def grouped_hamiltonian_analysis(
    groups: str,
    expectation_values: Optional[str] = None,
):
    """Calculates the number of measurements required for computing
    the expectation value of a qubit hamiltonian, where co-measurable terms
    are grouped as a list of QubitOperators.

    We are assuming the exact expectation values are provided
    (i.e. infinite number of measurements or simulations without noise)
    M ~ (\sum_{i} prec(H_i)) ** 2.0 / (epsilon ** 2.0)
    where prec(H_i) is the precision (square root of the variance)
    for each group of co-measurable terms H_{i}. It is computed as
    prec(H_{i}) = \sum{ab} |h_{a}^{i}||h_{b}^{i}| cov(O_{a}^{i}, O_{b}^{i})
    where h_{a}^{i} is the coefficient of the a-th operator, O_{a}^{i}, in the
    i-th group. Covariances are assumed to be zero for a != b:
    cov(O_{a}^{i}, O_{b}^{i}) = <O_{a}^{i} O_{b}^{i}> - <O_{a}^{i}> <O_{b}^{i}> = 0

    ARGS:
        groups (str): The name of a file containing a list of QubitOperator objects,
            where each element in the list is a group of co-measurable terms.
        expectation_values (str): The name of a file containing a single ExpectationValues
            object with all expectation values of the operators contained in groups.
            If absent, variances are assumed to be maximal, i.e. 1.
            NOTE: YOU HAVE TO MAKE SURE THAT THE ORDER OF EXPECTATION VALUES MATCHES
            THE ORDER OF THE TERMS IN THE *GROUPED* TARGET QUBIT OPERATOR,
            OTHERWISE THIS FUNCTION WILL NOT RETURN THE CORRECT RESULT.
    """

    grouped_operator = load_qubit_operator_set(groups)

    if expectation_values is not None:
        expecval = load_expectation_values(expectation_values)
    else:
        expecval = None

    K_coeff, nterms, frame_meas = estimate_nmeas_for_frames(grouped_operator, expecval)

    save_nmeas_estimate(
        nmeas=K_coeff,
        nterms=nterms,
        frame_meas=frame_meas,
        filename="hamiltonian_analysis.json",
    )


def expectation_values_from_rdms(
    interactionrdm: str,
    qubit_operator: str,
    sort_terms: bool = False,
):
    operator = load_qubit_operator(qubit_operator)
    rdms = load_interaction_rdm(interactionrdm)
    expecval = get_expectation_values_from_rdms(rdms, operator, sort_terms=sort_terms)
    save_expectation_values(expecval, "expectation_values.json")


def expectation_values_from_rdms_for_qubitoperator_list(
    interactionrdm: str, qubit_operator_list: str, sort_terms: bool = False
):
    """Computes expectation values of Pauli strings in a list of QubitOperator given a
       fermionic InteractionRDM from OpenFermion. All the expectation values for the
       operators in the list are written to file in a single ExpectationValues object in the
       same order the operators came in.

    ARGS:
        interactionrdm (str): The name of the file containing the interaction RDM to
            use for the expectation values computation
        qubitoperator_list (str): The name of the file containing the list of qubit operators
            to compute the expectation values for in the form of OpenFermion QubitOperator objects
        sort_terms (bool): whether or not each input qubit operator needs to be sorted before
            calculating expectations
    """

    operator_list = load_qubit_operator_set(qubit_operator_list)
    rdms = load_interaction_rdm(interactionrdm)
    expecval = get_expectation_values_from_rdms_for_qubitoperator_list(
        rdms, operator_list, sort_terms=sort_terms
    )
    save_expectation_values(expecval, "expectation_values.json")
