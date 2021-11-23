import json
from typing import Dict, List, Optional, Union

import openfermion
from zquantum.core import circuits
from zquantum.core.circuits import layouts
from zquantum.core.cost_function import sum_expectation_values
from zquantum.core.distribution import save_measurement_outcome_distribution
from zquantum.core.estimation import estimate_expectation_values_by_averaging
from zquantum.core.hamiltonian import (
    estimate_nmeas_for_frames,
    get_expectation_values_from_rdms,
    get_expectation_values_from_rdms_for_qubitoperator_list,
)
from zquantum.core.measurement import (
    Measurements,
    load_expectation_values,
    save_expectation_values,
)
from zquantum.core.openfermion import (
    change_operator_type,
    load_interaction_rdm,
    load_qubit_operator,
    load_qubit_operator_set,
)
from zquantum.core.serialization import load_array
from zquantum.core.typing import Specs
from zquantum.core.utils import (
    create_object,
    load_noise_model,
    save_list,
    save_nmeas_estimate,
    save_value_estimate,
)


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
        backend_specs["device_connectivity"] = layouts.load_circuit_connectivity(
            device_connectivity
        )

    backend = create_object(backend_specs)
    if "track_measurements" in backend_specs & backend_specs["track_measurements"]:
        backend = backend._make_measurement_tracking_backend()

    if isinstance(circuit, str):
        circuit = circuits.load_circuit(circuit)
    else:
        circuit = circuits.circuit_from_dict(circuit)

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
        backend_specs["device_connectivity"] = layouts.load_circuit_connectivity(
            device_connectivity
        )

    circuit_set = circuits.load_circuitset(circuitset)
    backend = create_object(backend_specs)
    if "track_measurements" in backend_specs & backend_specs["track_measurements"]:
        backend = backend._make_measurement_tracking_backend()

    n_samples_list = [n_samples for _ in circuit_set]
    measurements_set = backend.run_circuitset_and_measure(
        circuit_set, n_samples=n_samples_list
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
        backend_specs["device_connectivity"] = layouts.load_circuit_connectivity(
            device_connectivity
        )
    backend = create_object(backend_specs)
    if "track_measurements" in backend_specs & backend_specs["track_measurements"]:
        backend = backend._make_measurement_tracking_backend()

    circuit = circuits.load_circuit(circuit)

    bitstring_distribution = backend.get_bitstring_distribution(circuit)
    save_measurement_outcome_distribution(
        bitstring_distribution, "bitstring-distribution.json"
    )


def evaluate_ansatz_based_cost_function(
    ansatz_specs: Specs,
    backend_specs: Specs,
    cost_function_specs: Specs,
    ansatz_parameters: str,
    target_operator: Union[str, openfermion.SymbolicOperator],
    estimation_method_specs: Optional[Specs] = None,
    estimation_preprocessors_specs: Optional[List[Specs]] = None,
    noise_model: Optional[str] = None,
    device_connectivity: Optional[str] = None,
    prior_expectation_values: Optional[str] = None,
    estimation_tasks_transformations_kwargs: Optional[Dict] = None,
):
    # Empty dict as default is bad
    if estimation_tasks_transformations_kwargs is None:
        estimation_tasks_transformations_kwargs = {}
    ansatz_parameters = load_array(ansatz_parameters)
    # Load qubit op
    if isinstance(target_operator, str):
        operator = load_qubit_operator(target_operator)
    else:
        operator = target_operator
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
        backend_specs["device_connectivity"] = layouts.load_circuit_connectivity(
            device_connectivity
        )

    backend = create_object(backend_specs)
    if "track_measurements" in backend_specs & backend_specs["track_measurements"]:
        backend = backend._make_measurement_tracking_backend()

    if isinstance(cost_function_specs, str):
        cost_function_specs = json.loads(cost_function_specs)

    if (
        "estimator-specs" in cost_function_specs.keys()
        or "estimation-tasks-transformations-specs" in cost_function_specs.keys()
    ):
        raise RuntimeError(
            "Estimation-related specs should be separate arguments and not in "
            "cost_function_specs"
        )

    if prior_expectation_values is not None:
        if isinstance(prior_expectation_values, str):
            prior_expectation_values = load_expectation_values(prior_expectation_values)

    if estimation_method_specs is not None:
        if isinstance(estimation_method_specs, str):
            estimation_method_specs = json.loads(estimation_method_specs)
        estimation_method = create_object(estimation_method_specs)
    else:
        estimation_method = estimate_expectation_values_by_averaging

    cost_function_specs["estimation_method"] = estimation_method

    if estimation_preprocessors_specs is not None:
        cost_function_specs["estimation_preprocessors"] = []
        for estimation_tasks_transformation_specs in estimation_preprocessors_specs:

            if isinstance(estimation_tasks_transformation_specs, str):
                estimation_tasks_transformation_specs = json.loads(
                    estimation_tasks_transformation_specs
                )

            if prior_expectation_values is not None:
                # Since we don't know which estimation task transformation uses
                # prior_expectation_values, we add it to the kwargs of each one. If not
                # used by a particular transformer, it will be ignored.
                estimation_tasks_transformation_specs[
                    "prior_expectation_values"
                ] = prior_expectation_values
            cost_function_specs["estimation_preprocessors"].append(
                create_object(
                    estimation_tasks_transformation_specs,
                    **estimation_tasks_transformations_kwargs
                )
            )

    # cost_function.estimator.prior_expectation_values
    cost_function_specs["target_operator"] = operator
    cost_function_specs["ansatz"] = ansatz
    cost_function_specs["backend"] = backend
    cost_function = create_object(cost_function_specs)

    value_estimate = cost_function(ansatz_parameters)

    save_value_estimate(value_estimate, "value_estimate.json")


def grouped_hamiltonian_analysis(
    groups: str,
    expectation_values: Optional[str] = None,
):
    """Calculates the number of measurements required for computing
    the expectation value of a qubit hamiltonian, where co-measurable terms
    are grouped as a list of QubitOperators.

    We are assuming the exact expectation values are provided
    (i.e. infinite number of measurements or simulations without noise)
    M ~ (\\sum_{i} prec(H_i)) ** 2.0 / (epsilon ** 2.0)
    where prec(H_i) is the precision (square root of the variance)
    for each group of co-measurable terms H_{i}. It is computed as
    prec(H_{i}) = \\sum{ab} |h_{a}^{i}||h_{b}^{i}| cov(O_{a}^{i}, O_{b}^{i})
    where h_{a}^{i} is the coefficient of the a-th operator, O_{a}^{i}, in the
    i-th group. Covariances are assumed to be zero for a != b:
    cov(O_{a}^{i}, O_{b}^{i}) = <O_{a}^{i} O_{b}^{i}> - <O_{a}^{i}> <O_{b}^{i}> = 0

    Args:
        groups: The name of a file containing a list of QubitOperator objects,
            where each element in the list is a group of co-measurable terms.
        expectation_values: The name of a file containing a single ExpectationValues
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
       operators in the list are written to file in a single ExpectationValues object in
       the same order the operators came in.

    Args:
        interactionrdm: The name of the file containing the interaction RDM to
            use for the expectation values computation
        qubitoperator_list: The name of the file containing the list of qubit operators
            to compute the expectation values for in the form of OpenFermion
            QubitOperator objects
        sort_terms: whether or not each input qubit operator needs to be sorted before
            calculating expectations
    """

    operator_list = load_qubit_operator_set(qubit_operator_list)
    rdms = load_interaction_rdm(interactionrdm)
    expecval = get_expectation_values_from_rdms_for_qubitoperator_list(
        rdms, operator_list, sort_terms=sort_terms
    )
    save_expectation_values(expecval, "expectation_values.json")


def get_summed_expectation_values(
    operator: str, measurements: str, use_bessel_correction: Optional[bool] = True
):
    if isinstance(operator, str):
        operator = load_qubit_operator(operator)
        operator = change_operator_type(operator, openfermion.IsingOperator)
    loaded_measurements: Measurements
    if isinstance(measurements, str):
        loaded_measurements = Measurements.load_from_file(measurements)
    else:
        loaded_measurements = measurements
    expectation_values = loaded_measurements.get_expectation_values(
        operator, use_bessel_correction=use_bessel_correction
    )
    value_estimate = sum_expectation_values(expectation_values)
    save_value_estimate(value_estimate, "value-estimate.json")
