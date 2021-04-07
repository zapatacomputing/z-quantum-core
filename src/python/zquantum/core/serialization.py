"""Serialization module."""
import json
from numbers import Number
from operator import attrgetter
from typing import Any, Iterator

import numpy as np
from scipy.optimize import OptimizeResult

from .bitstring_distribution import BitstringDistribution, is_bitstring_distribution
from .history.recorder import HistoryEntry, HistoryEntryWithArtifacts
from .interfaces.optimizer import optimization_result
from .typing import AnyPath
from .utils import SCHEMA_VERSION, ValueEstimate, convert_array_to_dict


def has_numerical_keys(dictionary):
    return all(isinstance(value, Number) for value in dictionary.values())


def preprocess(tree):
    """This inflates namedtuples into dictionaries, otherwise they would be serialized as lists.

    KJ: I found initial version of this code a while ago in a related SO question:
    https://stackoverflow.com/questions/43913256/understanding-subclassing-of-jsonencoder
    """
    if isinstance(tree, dict):
        preprocessed = {k: preprocess(v) for k, v in tree.items()}
        if isinstance(tree, OptimizeResult):
            preprocessed["schema"] = SCHEMA_VERSION + "-optimization_result"
        return preprocessed
    elif isinstance(tree, tuple) and hasattr(tree, "_asdict"):
        return preprocess(tree._asdict())
    elif isinstance(tree, ValueEstimate):
        return tree.to_dict()
    elif isinstance(tree, (list, tuple)):
        return list(map(preprocess, tree))
    return tree


class OrquestraEncoder(json.JSONEncoder):
    ENCODERS_TABLE = {
        np.ndarray: convert_array_to_dict,
        ValueEstimate: ValueEstimate.to_dict,
        BitstringDistribution: attrgetter("distribution_dict"),
    }

    def default(self, o: Any):
        if type(o) in self.ENCODERS_TABLE:
            return self.ENCODERS_TABLE[type(o)](o)
        return o

    def encode(self, o: Any):
        return super().encode(preprocess(o))

    def iterencode(self, o: Any, _one_shot: bool = ...) -> Iterator[str]:
        return super().iterencode(preprocess(o))


class OrquestraDecoder(json.JSONDecoder):
    """Custom decoder for loading data dumped by ZapataEncoder."""

    SCHEMA_MAP = {
        "zapata-v1-value_estimate": ValueEstimate.from_dict,
        "zapata-v1-optimization_result": lambda obj: optimization_result(**obj),
    }

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        # Parts of the below if-elif-else are sketchy, because for some objects there is
        # no defined schema and we are matching object's type based on deserialized
        # dict's contents.
        # TODO: reimplement this when we agree on some common serialization mechanism.
        # See: https://www.pivotaltracker.com/story/show/175446541
        if "real" in obj:
            array = np.array(obj["real"])
            if "imag" in obj:
                array = array + 1j * np.array(obj["imag"])
            return array
        elif "call_number" in obj and "value" in obj:
            cls = HistoryEntry if "artifacts" not in obj else HistoryEntryWithArtifacts
            return cls(**obj)
        elif has_numerical_keys(obj) and is_bitstring_distribution(obj):
            return BitstringDistribution(obj, normalize=False)
        elif "schema" in obj and obj["schema"] in self.SCHEMA_MAP:
            return self.SCHEMA_MAP[obj.pop("schema")](obj)
        else:
            return obj


def save_optimization_results(optimization_results: dict, filename: AnyPath):
    optimization_results["schema"] = SCHEMA_VERSION + "-optimization_result"
    with open(filename, "wt") as target_file:
        json.dump(optimization_results, target_file, cls=OrquestraEncoder)


def load_optimization_results(filename: AnyPath):
    with open(filename) as source_file:
        return json.load(source_file, cls=OrquestraDecoder)
