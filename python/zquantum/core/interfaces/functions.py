"""Protocols describing different kinds of functions."""
from typing import NamedTuple, Callable, Any, TypeVar, Optional
from typing_extensions import Protocol, runtime_checkable
import numpy as np

T = TypeVar("T", covariant=True)
S = TypeVar("S", contravariant=True)


class StoreArtifact(Protocol):
    """A protocol describing how the artifacts are stored."""
    def __call__(self, artifact_name: str, artifact: Any, force: bool = False) -> None:
        pass


@runtime_checkable
class CallableWithGradient(Protocol):
    """A callable with gradient."""

    def __call__(self, params: np.ndarray):
        pass

    def gradient(self, params: np.ndarray) -> np.ndarray:
        pass


@runtime_checkable
class CallableStoringArtifacts(Protocol[S, T]):
    """A callable that stores artifacts."""

    def __call__(
        self, params: S, store_artifact: Optional[StoreArtifact]
    ) -> T:
        pass


@runtime_checkable
class CallableWithGradientStoringArtifacts(
    CallableStoringArtifacts[np.ndarray, T], Protocol
):
    """A callable with gradient that stored artifacts."""
    def gradient(self, params: np.ndarray) -> np.ndarray:
        pass


class FunctionWithGradient(NamedTuple):
    """A callable with gradient."""
    function: Callable[[np.ndarray], np.ndarray]
    gradient: Callable[[np.ndarray], np.ndarray]

    def __call__(self, params: np.ndarray):
        return self.function(params)
