"""Types commonly encountered in zquantum repositories."""
from os import PathLike
from typing import Dict, Union

from typing_extensions import Protocol


class Readable(Protocol):
    def read(self) -> str:
        pass


class Writeable(Protocol):
    def write(self, content: str):
        pass


AnyPath = Union[str, bytes, PathLike]

LoadSource = Union[Readable, AnyPath]

DumpTarget = Union[Writeable, AnyPath]

Specs = Union[str, Dict]
