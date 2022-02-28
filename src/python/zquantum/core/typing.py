"""Types commonly encountered in zquantum repositories."""
from numbers import Number
from os import PathLike
from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np
import sympy
from typing_extensions import Protocol, runtime_checkable

from .history.recorder import (
    ArtifactRecorder,
    HistoryEntry,
    HistoryEntryWithArtifacts,
    SimpleRecorder,
)


@runtime_checkable
class Readable(Protocol):
    def read(self, size: int = 0) -> str:
        pass

    def writable(self) -> bool:
        pass


@runtime_checkable
class Writeable(Protocol):
    def write(self, content: str):
        pass

    def writable(self) -> bool:
        pass


AnyPath = Union[str, bytes, PathLike]

LoadSource = Union[Readable, AnyPath]

DumpTarget = Union[Writeable, AnyPath]

Specs = Union[str, Dict]

AnyRecorder = Union[SimpleRecorder, ArtifactRecorder]
AnyHistory = Union[List[HistoryEntry], List[HistoryEntryWithArtifacts]]
RecorderFactory = Callable[[Callable], AnyRecorder]


Parameter = Union[sympy.Symbol, Number]
ParameterizedVector = Union[Sequence[Parameter], np.ndarray]


class SupportsLessThan(Protocol):
    def __lt__(self, other: Any) -> bool:
        """Return result of comparison self < other."""
