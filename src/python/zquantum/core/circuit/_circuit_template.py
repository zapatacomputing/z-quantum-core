import json
import numpy as np
import importlib
from numpy.random import random_sample
from ..utils import SCHEMA_VERSION
from ..utils import convert_array_to_dict, convert_dict_to_array
from ..circuit import Circuit, Qubit, Gate
from typing import TextIO, List, Tuple, Dict, Optional
from scipy.optimize import OptimizeResult


def save_circuit_template(circuit_template: dict, filename: str):
    """Saves a circuit template to a file.

    Args:
        circuit_template (dict): the circuit template to be saved
        filename (str): the name of the file
    """

    try:
        circuit_template["ansatz_kwargs"]["layers"] = circuit_template["ansatz_kwargs"][
            "layers"
        ].to_dict()
    except KeyError:
        pass
    circuit_template = dict(circuit_template)
    circuit_template["schema"] = SCHEMA_VERSION + "-circuit_template"
    with open(filename, "w") as f:
        f.write(json.dumps(circuit_template))


def load_circuit_template(file: TextIO) -> dict:
    """Loads a circuit template from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        dict: the circuit template
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    try:
        data["ansatz_kwargs"]["layers"] = CircuitLayers.from_dict(
            data["ansatz_kwargs"]["layers"]
        )
    except KeyError:
        pass

    return data


def save_circuit_template_params(params: np.ndarray, filename: str) -> None:
    """Saves a circuit object to a file.

    Args:
        params (numpy.ndarray): the parameters to be saved
        filename (str): the name of the file
    """

    dictionary = {"schema": SCHEMA_VERSION + "-circuit_template_params"}
    dictionary["parameters"] = convert_array_to_dict(params)
    with open(filename, "w") as f:
        f.write(json.dumps(dictionary))


def load_circuit_template_params(file: TextIO):
    """Loads a circuit template from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        dict: the circuit template
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return convert_dict_to_array(data["parameters"])


def build_ansatz_circuit(ansatz: dict, params: np.ndarray) -> Circuit:
    """Construct the circuit corresponding to the ansatz.

    Args:
        ansatz (dict): the ansatz
        params (numpy.ndarray): the ansatz parameters

    Returns:
        core.circuit.Circuit: the circuit
    """

    module = importlib.import_module(ansatz["ansatz_module"])
    func = getattr(module, ansatz["ansatz_func"])
    qprog = func(params, **ansatz["ansatz_kwargs"])

    return qprog


def combine_ansatz_params(params1: np.ndarray, params2: np.ndarray) -> np.ndarray:
    """Combine two sets of ansatz parameters.
    
    Args:
        params1 (numpy.ndarray): the first set of parameters
        params2 (numpy.ndarray): the second set of parameters
    
    Returns:
        numpy.ndarray: the combined parameters
    """
    return np.concatenate((params1, params2))


class ParameterGrid:
    """A class representing a grid of parameter values to be used in a grid search.

    Args:
        param_ranges (list): ranges of the parameters describing the shape of the grid. Each range consist is of the form (min, max, step).

    Attributes:
        param_ranges (list): same as above.
    """

    def __init__(self, param_ranges: List[Tuple[float]]):
        self.param_ranges = param_ranges

    @property
    def params_list(self) -> np.ndarray:
        grid_array = np.reshape(np.stack(self.params_meshgrid), (self.n_params, -1))

        grid = []
        for i in range(grid_array.shape[1]):
            grid.append(grid_array[:, i].flatten())

        return grid

    def to_dict(self) -> dict:
        return {"param_ranges": self.param_ranges}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data["param_ranges"])

    @property
    def params_meshgrid(self) -> np.ndarray:
        """
        Creates a meshgrid from the parameter ranges.
        """
        param_vectors = []

        for param_spec in self.param_ranges:
            param_vectors.append(np.arange(param_spec[0], param_spec[1], param_spec[2]))

        return np.meshgrid(*param_vectors, indexing="ij")

    @property
    def n_params(self) -> int:
        return len(self.param_ranges)


def save_parameter_grid(grid: ParameterGrid, filename: str) -> None:
    """Saves a parameter grid to a file.

    Args:
        grid (core.circuit.ParameterGrid): the parameter grid to be saved
        filename (str): the name of the file
    """

    data = grid.to_dict()
    data["schema"] = SCHEMA_VERSION + "-parameter_grid"

    with open(filename, "w") as f:
        f.write(json.dumps(data))


def load_parameter_grid(file: TextIO) -> ParameterGrid:
    """Loads a parameter grid from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        core.circuit.ParameterGrid: the parameter grid
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return ParameterGrid.from_dict(data)


def build_uniform_param_grid(
    n_params_per_layer: int,
    n_layers: int = 1,
    min_value: float = 0.0,
    max_value: float = 2 * np.pi,
    step: float = np.pi / 5,
) -> ParameterGrid:
    """Builds a uniform grid of parameters.

    Args:
        n_params_per_layer (int): number of parameters for each layer
        n_layers (int): the number of layers to create parameters for
        min_value (float): the minimum value for the parameters
        max_value (float): the maximum value for the parameters
        step (float): the step size
    
    Returns:
        list: a list of numpy.ndarray objects representing points on a grid in parameter space
    """

    n_params = n_params_per_layer * n_layers

    param_ranges = [(min_value, max_value, step)] * n_params
    return ParameterGrid(param_ranges)


class CircuitLayers(object):
    """A class representing a pattern of circuit layers, consisting of lists,
        each list containing the groups of qubits entangled for each multiqubit
        gate in a particular layer.
    """

    def __init__(self, layers: List[List[Tuple]]):
        """
        Args:
            layers: list of list of tuples, each tuple
            representing a group of qubits that is connected in the layer by a 
            multiqubit gate.
        """
        self.layers = layers

    def to_dict(self) -> dict:
        return {"layers": self.layers}

    @classmethod
    def from_dict(cls, data: Dict[str, List[List[Tuple]]]):
        layers = []
        for layer in data["layers"]:
            layers.append([tuple(x) for x in layer])
        return cls(layers)


def save_circuit_layers(circuit_layers: CircuitLayers, filename: str) -> None:
    """Saves a list of circuit layers to a file.
    Args:
        circuit_layers (circuit.CircuitLayers)
        filename (str): the name of the file
    """

    circuit_layers = circuit_layers.to_dict()
    circuit_layers["schema"] = SCHEMA_VERSION + "-circuit_layers"
    with open(filename, "w") as f:
        f.write(json.dumps(circuit_layers))


def load_circuit_layers(file: TextIO) -> CircuitLayers:
    """Loads a list of circuit layers from a file.
    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        (circuit.CircuitLayers)
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return CircuitLayers.from_dict(data)


def save_circuit_ordering(ordering, filename):
    """Saves a circuit ordering (e.g. mapping from spin-orbitals to qubits) to a file.
    Args:
        ordering (list)
        filename (str): the name of the file
    """

    ordering = {"ordering": ordering}
    ordering["schema"] = SCHEMA_VERSION + "-circuit_ordering"
    with open(filename, "w") as f:
        f.write(json.dumps(ordering))


def load_circuit_ordering(file):
    """Loads a circuit ordering (e.g. mapping from spin-orbitals to qubits) to a file.
    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        ordering (list)
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return data["ordering"]


class CircuitConnectivity(object):
    """A class representing the connectivity of a circuit resulting from qpu
        constraints, consisting of a list of tuples of qubits representing the 
        allowed multiqubit gate connections.
    """

    def __init__(self, connections):
        """
        Args:
            connections: list of tuples representing groups of qubits
        """
        self.connectivity = connections

    def to_dict(self):
        return {"connectivity": self.connectivity}

    @classmethod
    def from_dict(cls, data):
        tuples = [tuple(x) for x in data["connectivity"]]
        return cls(tuples)


def save_circuit_connectivity(circuit_connectivity, filename):
    """Saves a circuit connectivity to a file.
    Args:
        circuit_connectivity (zquantum.core.circuit.CircuitConnectivity)
        filename (str): the name of the file
    """

    circuit_connectivity = circuit_connectivity.to_dict()
    circuit_connectivity["schema"] = SCHEMA_VERSION + "-circuit_connectivity"
    with open(filename, "w") as f:
        f.write(json.dumps(circuit_connectivity))


def load_circuit_connectivity(file):
    """Loads a circuit connectivity from a file.
    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        (zquantum.core.circuit.CircuitConnectivity)
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return CircuitConnectivity.from_dict(data)


def build_circuit_layers_and_connectivity(
    x_dimension, y_dimension=None, layer_type="nearest-neighbor"
):
    """ Function to generate circuit layers for 1-dimensional and 2-dimensional
    arrays of qubits
    Args:
        x_dimension (int): number of qubits per row of the array
        y_dimension (int): number of qubits per column of the array
        layer_type (str): string designating type of layer to be created
    Returns:
        (zquantum.core.circuit.CircuitConnectivity, zquantum.core.circuit.CircuitLayers)
    """
    if layer_type == "sycamore":
        return _build_circuit_layers_and_connectivity_sycamore(x_dimension, y_dimension)
    elif layer_type == "nearest-neighbor":
        return _build_circuit_layers_and_connectivity_nearest_neighbors(x_dimension)
    else:
        ValueError("Layer type {0} is not defined".format(layer_type))


def _build_circuit_layers_and_connectivity_sycamore(x_dimension, y_dimension):
    """ Function to generate circuit connectivity and circuit layers
        for 2D quantum processors with sycamore-like connectivity
    Args:
        x_dimension (int): number of qubits per row of the array
        y_dimension (int): number of qubits per column of the array
    Returns:
        (zquantum.core.circuit.CircuitConnectivity, zquantum.core.circuit.CircuitLayers)
    """
    connectivity = []

    # two-colors patterns: patterns
    pattern1 = np.zeros((y_dimension - 1, 2 * x_dimension - 1))
    pattern2 = np.zeros((y_dimension - 1, 2 * x_dimension - 1))

    for m in range(0, y_dimension - 1):
        pattern2[m, :] = m % 2

    py_index = 0
    for m, y_index in enumerate(range(1, y_dimension, 2)):
        row_up = []
        row_down = []
        px_index = 0
        for x_index in range(0, x_dimension):
            node = y_index * x_dimension + x_index
            if x_index == x_dimension - 1:
                row_up.append((node, node - x_dimension))
                pattern1[py_index, px_index] = (x_index + m) % 2
                if y_dimension - 1 > y_index:
                    row_down.append((node, node + x_dimension))
                    pattern1[py_index + 1, px_index] = (x_index + m + 1) % 2
                px_index += 1
            else:
                row_up.extend(
                    [(node, node - x_dimension), (node, node - x_dimension + 1)]
                )
                pattern1[py_index, px_index] = (x_index + m) % 2
                pattern1[py_index, px_index + 1] = (x_index + m) % 2
                if y_dimension - 1 > y_index:
                    row_down.extend(
                        [(node, node + x_dimension), (node, node + x_dimension + 1)]
                    )
                    pattern1[py_index + 1, px_index] = (x_index + m + 1) % 2
                    pattern1[py_index + 1, px_index + 1] = (x_index + m + 1) % 2
                px_index += 2
        py_index += 2

        connectivity.append(row_up)
        connectivity.append(row_down)

    # three-color patterns: masks
    masks = []
    for n in range(0, 2):
        mask1 = pattern1.copy()
        mask2 = pattern2.copy()
        for m in range(mask1.shape[0]):
            mask1[m, (m + n) % 2 : mask1.shape[1] : 2] = -1
            mask2[m, (m + n) % 2 : mask1.shape[1] : 2] = -1
        masks.extend([mask1, mask2])

    # generate layers
    layers = []
    for n, mask in enumerate(masks):
        layer1 = []
        layer2 = []
        for x in range(mask.shape[1]):
            for y in range(mask.shape[0]):
                if mask[y, x] == 0:
                    layer1.append(connectivity[y][x])
                elif mask[y, x] == 1:
                    layer2.append(connectivity[y][x])
        layers.extend([layer1, layer2])

    final_connectivity = []
    for group in connectivity:
        final_connectivity.extend(group)

    return CircuitConnectivity(final_connectivity), CircuitLayers(layers)


def _build_circuit_layers_and_connectivity_nearest_neighbors(n_qubits):
    """ Function to generate circuit layers for processors with nearest-neighbor 
    connectivity
    Args:
        n_qubits (int): number of qubits in the qubit array
    Returns:
        (zquantum.core.circuit.CircuitConnectivity, zquantum.core.circuit.CircuitLayers)
    """
    even_layer = []
    odd_layer = []
    for index in range(0, n_qubits - 1, 2):
        even_layer.append((index, index + 1))
    for index in range(1, n_qubits - 1, 2):
        odd_layer.append((index, index + 1))
    connectivity = []
    connectivity.extend(even_layer)
    connectivity.extend(odd_layer)
    return CircuitConnectivity(connectivity), CircuitLayers([even_layer, odd_layer])


def create_layer_of_gates(
    number_of_qubits: int, gate_name: str, parameters: Optional[np.ndarray] = None
) -> Circuit:
    """ Creates a circuit consisting of a layer of single-qubit gates acting on all qubits.

    Args:
        number_of_qubits (int): number of qubits in the circuit
        gate_name (str): the single qubit gate to be applied to each qubit
        params (numpy.array): parameters of the single-qubit gates

    Returns:
        Circuit: a zquantum.core.circuit.Circuit object
    """
    circuit = Circuit()
    circuit.qubits = [Qubit(i) for i in range(number_of_qubits)]

    if parameters is not None:
        assert len(parameters) == number_of_qubits

        circuit.gates = [
            Gate(gate_name, [circuit.qubits[i]], [parameters[i]])
            for i in range(number_of_qubits)
        ]
    else:
        circuit.gates = [
            Gate(gate_name, [circuit.qubits[i]]) for i in range(number_of_qubits)
        ]
    return circuit
