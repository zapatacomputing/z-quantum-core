import json
import numpy as np
import importlib
from numpy.random import random_sample
from ..utils import SCHEMA_VERSION
from ..utils import convert_array_to_dict, convert_dict_to_array
from ..circuit import Circuit
from typing import TextIO, List, Tuple, Dict
from scipy.optimize import OptimizeResult

def save_circuit_template(circuit_template: dict, filename: str):
    """Saves a circuit template to a file.

    Args:
        circuit_template (dict): the circuit template to be saved
        filename (str): the name of the file
    """

    try:
        circuit_template['ansatz_kwargs']['layers'] = circuit_template['ansatz_kwargs']['layers'].to_dict()
    except KeyError:
        pass
    circuit_template = dict(circuit_template)
    circuit_template['schema'] = SCHEMA_VERSION+'-circuit_template'
    with open(filename, 'w') as f:
        f.write(json.dumps(circuit_template))

def load_circuit_template(file: TextIO) -> dict:
    """Loads a circuit template from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        dict: the circuit template
    """

    if isinstance(file, str):
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        data = json.load(file)

    try:
        data['ansatz_kwargs']['layers'] = CircuitLayers.from_dict(data['ansatz_kwargs']['layers'])
    except KeyError:
        pass

    return(data)

def save_circuit_template_params(params: np.ndarray, filename: str) -> None:
    """Saves a circuit object to a file.

    Args:
        params (numpy.ndarray): the parameters to be saved
        filename (str): the name of the file
    """

    dictionary = {'schema': SCHEMA_VERSION + '-circuit_template_params'}
    dictionary['parameters'] = convert_array_to_dict(params)
    with open(filename, 'w') as f:
        f.write(json.dumps(dictionary))

def load_circuit_template_params(file: TextIO):
    """Loads a circuit template from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        dict: the circuit template
    """

    if isinstance(file, str):
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return convert_dict_to_array(data['parameters'])

def build_ansatz_circuit(ansatz: dict, params: np.ndarray) -> Circuit:
    """Construct the circuit corresponding to the ansatz.

    Args:
        ansatz (dict): the ansatz
        params (numpy.ndarray): the ansatz parameters

    Returns:
        core.circuit.Circuit: the circuit
    """

    module = importlib.import_module(ansatz['ansatz_module'])
    func = getattr(module, ansatz['ansatz_func'])
    qprog = func(params, **ansatz['ansatz_kwargs'])

    return qprog

def generate_random_ansatz_params(ansatz: dict, 
                                min_val: float=0, 
                                max_val: float=1., 
                                n_layers: int=1,
                                include_non_layered_params: bool=True, 
                                layer_index: int=0) -> np.ndarray:
    """For the given ansatz, generate random parameters.

    Args:
        ansatz (dict): the ansatz
        min_val (float): minimum parameter value
        max_val (float): maximum parameter value
        n_layers (int): number of layers for params with a
            layer-by-layer structured
        include_non_layered_params (bool): whether non_layered_params
            are considered in the calculation
        layer_index (int): for ansatz with sublayers, the index of the sublayer where
                        to start the guess
    Returns:
        numpy.ndarray: the generated parameters
    """
    n_params = 0
    for i in range(n_layers):
        n_params += ansatz['n_params'][(i + layer_index) % len(ansatz['n_params'])]
        if 'ansatz_type' in ansatz.keys():
            if 'IBM' in ansatz['ansatz_type'] and 'HEA v2' in ansatz['ansatz_type']:
                if (i + layer_index) % len(ansatz['n_params']) == 0:
                    n_params += 2 * ansatz['ansatz_kwargs']['n_mo']

    if ansatz.get('n_non_layered_params') and include_non_layered_params:
        n_params += ansatz['n_non_layered_params']
    params = (max_val - min_val) * random_sample(n_params) + min_val
    return params

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

    def __init__(self, param_ranges:List[Tuple[float]]):
        self.param_ranges = param_ranges

    @property
    def params_list(self) -> np.ndarray:
        grid_array = np.reshape(np.stack(self.params_meshgrid), (self.n_params, -1))

        grid = []
        for i in range(grid_array.shape[1]):
            grid.append(grid_array[:, i].flatten())
        
        return grid

    def to_dict(self) -> dict:
        return {'param_ranges': self.param_ranges}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data['param_ranges'])

    @property
    def params_meshgrid(self) -> np.ndarray:
        """
        Creates a meshgrid from the parameter ranges.
        """
        param_vectors = []
        
        for param_spec in self.param_ranges:
            param_vectors.append(np.arange(param_spec[0], param_spec[1], param_spec[2]))

        return np.meshgrid(*param_vectors, indexing='ij')

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
    data['schema'] =  SCHEMA_VERSION + '-parameter_grid'

    with open(filename, 'w') as f:
        f.write(json.dumps(data))

def load_parameter_grid(file: TextIO) -> ParameterGrid:
    """Loads a parameter grid from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        core.circuit.ParameterGrid: the parameter grid
    """

    if isinstance(file, str):
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return ParameterGrid.from_dict(data)

def build_uniform_param_grid(ansatz: dict, 
                            n_layers: int=1, 
                            min_value: float=0., 
                            max_value: float=2*np.pi, 
                            step: float=np.pi/5) -> ParameterGrid:
    """Builds a uniform grid of parameters.

    Args:
        ansatz (dict): a dict representing a variational circuit template
        n_layers (int): the number of layers to create parameters for
        min_value (float): the minimum value for the parameters
        max_value (float): the maximum value for the parameters
        step (float): the step size
    
    Returns:
        list: a list of numpy.ndarray objects representing points on a grid in parameter space
    """
    n_params = 0
    for i in range(n_layers):
        n_params += ansatz['n_params'][i % len(ansatz['n_params'])]

    param_ranges = [(min_value, max_value, step)]*n_params
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
        return {'layers': self.layers}

    @classmethod
    def from_dict(cls, data: Dict[str, List[List[Tuple]]]):
        layers = []
        for layer in data['layers']:
            layers.append([tuple(x) for x in layer])
        return cls(layers)

def save_circuit_layers(circuit_layers: CircuitLayers, filename: str) -> None:
    """Saves a list of circuit layers to a file.
    Args:
        circuit_layers (circuit.CircuitLayers)
        filename (str): the name of the file
    """

    circuit_layers = circuit_layers.to_dict()
    circuit_layers['schema'] = SCHEMA_VERSION+'-circuit_layers'
    with open(filename, 'w') as f:
        f.write(json.dumps(circuit_layers))

def load_circuit_layers(file: TextIO) -> CircuitLayers:
    """Loads a list of circuit layers from a file.
    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        (circuit.CircuitLayers)
    """

    if isinstance(file, str):
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return CircuitLayers.from_dict(data)

def save_circuit_ordering(ordering, filename):
    """Saves a circuit ordering (mapping from spin-orbitals to qubits) to a file.
    Args:
        ordering (list)
        filename (str): the name of the file
    """

    ordering = {'ordering': ordering}
    ordering['schema'] = SCHEMA_VERSION+'-circuit_ordering'
    with open(filename, 'w') as f:
        f.write(json.dumps(ordering))

def load_circuit_ordering(file):
    """Loads a circuit ordering (mapping from spin-orbitals to qubits) to a file.
    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        ordering (list)
    """

    if isinstance(file, str):
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return data['ordering']