import json
from zquantum.core.utils import create_object, load_noise_model, save_value_estimate, save_nmeas_estimate
from zquantum.core.measurement import load_expectation_values, save_expectation_values
from zquantum.core.hamiltonian import estimate_nmeas, get_expectation_values_from_rdms
from zquantum.core.circuit import (
    load_circuit,
    load_circuit_connectivity,
    load_circuit_template_params,
)
from zquantum.core.bitstring_distribution import save_bitstring_distribution
from qeopenfermion import load_qubit_operator, load_interaction_rdm
from typing import Dict


def run_circuit_and_measure(
    backend_specs: Dict,
    circuit: str,
    noise_model: str = "None",
    device_connectivity: str = "None",
):
    backend_specs = json.loads(backend_specs)
    if noise_model != "None":
        backend_specs["noise_model"] = load_noise_model(noise_model)
    if device_connectivity != "None":
        backend_specs["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )

    backend = create_object(backend_specs)
    circuit = load_circuit(circuit)

    measurements = backend.run_circuit_and_measure(circuit)
    measurements.save("measurements.json")


def get_bitstring_distribution(
    backend_specs: Dict,
    circuit: str,
    noise_model: str = "None",
    device_connectivity: str = "None",
):
    backend_specs = json.loads(backend_specs)
    if noise_model != "None":
        backend_specs["noise_model"] = load_noise_model(noise_model)
    if device_connectivity != "None":
        backend_specs["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )

    backend = create_object(backend_specs)
    circuit = load_circuit(circuit)

    bitstring_distribution = backend.get_bitstring_distribution(circuit)
    save_bitstring_distribution(bitstring_distribution, "bitstring-distribution.json")


def evaluate_ansatz_based_cost_function(
    ansatz_specs: str,
    backend_specs: str,
    cost_function_specs: str,
    ansatz_parameters: str,
    qubit_operator: str,
    noise_model="None",
    device_connectivity="None",
):
    ansatz_parameters = load_circuit_template_params(ansatz_parameters)
    # Load qubit op
    operator = load_qubit_operator(qubit_operator)
    ansatz_specs = json.loads(ansatz_specs)
    if ansatz_specs["function_name"] == "QAOAFarhiAnsatz":
        ansatz = create_object(ansatz_specs, cost_hamiltonian=operator)
    else:
        ansatz = create_object(ansatz_specs)

    backend_specs = json.loads(backend_specs)
    if noise_model != "None":
        backend_specs["noise_model"] = load_noise_model(noise_model)
    if device_connectivity != "None":
        backend_specs["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )

    backend = create_object(backend_specs)
    cost_function_specs = json.loads(cost_function_specs)
    estimator_specs = cost_function_specs.pop("estimator-specs", None)
    if estimator_specs is not None:
        cost_function_specs["estimator"] = create_object(estimator_specs)
    cost_function_specs["target_operator"] = operator
    cost_function_specs["ansatz"] = ansatz
    cost_function_specs["backend"] = backend
    cost_function = create_object(cost_function_specs)

    value_estimate = cost_function(ansatz_parameters)

    save_value_estimate(value_estimate, "value_estimate.json")

def hamiltonian_analysis(
    qubit_operator: str,
    decomposition_method: str = "greedy",
    expectation_values: str = "None",
):
    operator = load_qubit_operator(qubit_operator)
    if decomposition_method != "greedy-sorted" and decomposition_method != "greedy":
        raise ValueError(f'Decomposition method {decomposition_method} is not supported')
    if expectation_values != "None":
        expecval = load_expectation_values(expectation_values)
    else:
        expecval = None

    K_coeff, nterms, frame_meas = estimate_nmeas(operator, decomposition_method, expecval)
    save_nmeas_estimate(nmeas=K_coeff, nterms=nterms, frame_meas=frame_meas, filename='hamiltonian_analysis.json')
   
def expectation_values_from_rdms(
    interactionrdm: str,
    qubit_operator: str,
    sort_terms: bool = False,
):
    operator = load_qubit_operator(qubit_operator)
    rdms = load_interaction_rdm(interactionrdm)
    expecval = get_expectation_values_from_rdms(rdms, operator, sort_terms=sort_terms)
    save_expectation_values(expecval, 'expectation_values.json')
