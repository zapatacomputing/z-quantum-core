"""Serialization module."""
import json
from typing import Any, NamedTuple
import numpy as np
from .history.recorder import HistoryEntry, HistoryEntryWithArtifacts
from .utils import convert_array_to_dict, ValueEstimate


def history_entry_to_dict(entry: HistoryEntry):
    return {
        "call_number": entry.call_number,
        "params": convert_array_to_dict(entry.params),
        "value": entry.value,
    }


def history_entry_with_artifacts_to_dict(entry: HistoryEntryWithArtifacts):
    return {
        "call_number": entry.call_number,
        "params": convert_array_to_dict(entry.params),
        "value": entry.value,
        "artifacts": entry.artifacts,
    }


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
        HistoryEntry: history_entry_to_dict,
        np.ndarray: convert_array_to_dict,
        ValueEstimate: ValueEstimate.to_dict,
    }

    def default(self, o: Any):
        if type(o) in self.ENCODERS_TABLE:
            return self.ENCODERS_TABLE[type(o)](o)
        return o

    def encode(self, o: Any):
        return super().encode(preprocess(o))


def save_optimization_results(optimization_results, filename):
    with open(filename, "wt") as target_file:
        json.dump(optimization_results, target_file, cls=ZapataEncoder)
