"""Serialization module."""
import json
import os
from contextlib import contextmanager
from numbers import Number
from operator import attrgetter
from typing import Any, Callable, Dict, Iterator, Union

import numpy as np
from scipy.optimize import OptimizeResult

from .bitstring_distribution import BitstringDistribution, is_bitstring_distribution
from .history.recorder import HistoryEntry, HistoryEntryWithArtifacts
from .interfaces.optimizer import optimization_result
from .typing import AnyPath, DumpTarget, LoadSource
from .utils import (
    SCHEMA_VERSION,
    ValueEstimate,
    convert_array_to_dict,
    convert_dict_to_array,
)


def has_numerical_keys(dictionary):
    return all(isinstance(value, Number) for value in dictionary.values())


def preprocess(tree):
    """This inflates namedtuples into dictionaries, otherwise they would be serialized
    as lists.

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
    ENCODERS_TABLE: Dict[Any, Callable[[Any], Any]] = {
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

    def iterencode(self, o: Any, _one_shot: bool = False) -> Iterator[str]:
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
    with open(filename, "rt") as source_file:
        return json.load(source_file, cls=OrquestraDecoder)


@contextmanager
def ensure_open(path_like: Union[LoadSource, DumpTarget], mode="r", encoding="utf-8"):
    # str | bytes | PathLike | Readable
    if isinstance(path_like, (str, bytes, os.PathLike)):
        with open(path_like, mode, encoding=encoding if "b" not in mode else None) as f:
            yield f
    else:
        # Readable | Writable
        if set(mode).intersection(set("wxa+")) and not path_like.writable():
            raise ValueError(f"File isn't writable, can't ensure mode {mode}")
        yield path_like


ARRAY_SCHEMA = SCHEMA_VERSION + "-array"


def save_array(array: np.ndarray, path_like: DumpTarget) -> None:
    """Saves array to a file.

    Args:
        array : the parameters to be saved
        filename: the name of the file
    """

    dictionary: Dict[str, Any] = {"schema": ARRAY_SCHEMA}
    dictionary["array"] = convert_array_to_dict(array)
    with ensure_open(path_like, "w") as f:
        f.write(json.dumps(dictionary))


def load_array(file: LoadSource):
    """Loads array from a file.

    Args:
        file: the name of the file, or a file-like object.

    Returns:
        dict: the circuit template
    """

    with ensure_open(file, "r") as f:
        data = json.load(f)

    return convert_dict_to_array(data["array"])
