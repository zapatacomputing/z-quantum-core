"""Types commonly encountered in zquantum repositories."""
from abc import abstractmethod
from os import PathLike
from typing import Any, Callable, Dict, Union

from typing_extensions import Protocol

from .history.recorder import ArtifactRecorder, SimpleRecorder


class Readable(Protocol):
    def read(self) -> str:
        pass

    def writable(self) -> bool:
        pass


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
RecorderFactory = Callable[[Callable], AnyRecorder]


class SupportsLessThan(Protocol):
    def __lt__(self, other: Any) -> bool:
        """Return result of comparison self < other."""
