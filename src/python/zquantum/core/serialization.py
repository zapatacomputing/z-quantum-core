"""Serialization module."""
import json
from typing import Any, NamedTuple, Iterator
import numpy as np
from .utils import convert_array_to_dict, ValueEstimate


def preprocess(tree):
    """This inflates namedtuples into dictionaries, otherwise they would be serialized as lists.

    KJ: I found initial version of this code a while ago in a related SO question:
    https://stackoverflow.com/questions/43913256/understanding-subclassing-of-jsonencoder
    """
    if isinstance(tree, dict):
        return {k: preprocess(v) for k, v in tree.items()}
    elif isinstance(tree, tuple) and hasattr(tree, "_asdict"):
        return preprocess(tree._asdict())
    # Note: isinstance check with ValueEstimate is broken, that's why we compare types here
    elif type(tree) == ValueEstimate:
        return tree.to_dict()
    elif isinstance(tree, (list, tuple)):
        return list(map(preprocess, tree))
    return tree


class ZapataEncoder(json.JSONEncoder):
    ENCODERS_TABLE = {
        np.ndarray: convert_array_to_dict,
        ValueEstimate: ValueEstimate.to_dict,
    }

    def default(self, o: Any):
        if type(o) in self.ENCODERS_TABLE:
            return self.ENCODERS_TABLE[type(o)](o)
        return o

    def encode(self, o: Any):
        return super().encode(preprocess(o))

    def iterencode(self, o: Any, _one_shot: bool = ...) -> Iterator[str]:
        return super().iterencode(preprocess(o))


def save_optimization_results(optimization_results, filename):
    with open(filename, "wt") as target_file:
        json.dump(optimization_results, target_file, cls=ZapataEncoder)
