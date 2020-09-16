import json
from zquantum.core.utils import create_object, load_noise_model
from zquantum.core.circuit import load_circuit, load_circuit_connectivity
from zquantum.core.bitstring_distribution import save_bitstring_distribution


def run_circuit_and_measure(
    backend_specs, circuit, noise_model="None", device_connectivity="None"
):
    backend = create_object(json.loads(backend_specs))
    circuit = load_circuit(circuit)

    if noise_model != "None":
        backend_specs_dict["noise_model"] = load_noise_model(noise_model)
    if device_connectivity != "None":
        backend_specs_dict["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )
    measurements = backend.run_circuit_and_measure(circuit)
    measurements.save("measurements.json")


def get_bitstring_distribution(
    backend_specs, circuit, noise_model="None", device_connectivity="None"
):
    backend = create_object(json.loads(backend_specs))
    circuit = load_circuit(circuit)

    if noise_model != "None":
        backend_specs_dict["noise_model"] = load_noise_model(noise_model)
    if device_connectivity != "None":
        backend_specs_dict["device_connectivity"] = load_circuit_connectivity(
            device_connectivity
        )
    bitstring_distribution = backend.get_bitstring_distribution(circuit)
    save_bitstring_distribution(bitstring_distribution, "bitstring-distribution.json")
