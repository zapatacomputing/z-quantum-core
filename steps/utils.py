import numpy as np
import json
from typing import Union, Dict
from zquantum.core.utils import save_list, create_object

Specs = Union[str, Dict]


# Generate random list of integers
def generate_random_list_of_integers(seed: Union[None, int] = None, **kwargs):
    """Generate a random list of integers. Docs can be found at the link below, but the keyword arugment "low" is mandatory.
    https://numpy.org/devdocs/reference/random/generated/numpy.random.Generator.integers.html#numpy.random.Generator.integers
    """
    rng = np.random.default_rng(seed=seed)
    generated_integers = rng.integers(**kwargs).tolist()
    save_list(generated_integers, "integers.json")