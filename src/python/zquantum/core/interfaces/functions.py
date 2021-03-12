"""Protocols describing different kinds of functions."""
from inspect import signature
from typing import NamedTuple, Callable, Any, TypeVar, Optional, Union
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

    def __call__(self, params: np.ndarray) -> float:
        pass

    def gradient(self, params: np.ndarray) -> np.ndarray:
        pass


@runtime_checkable
class CallableStoringArtifacts(Protocol[S, T]):
    """A callable that stores artifacts."""

    def __call__(self, params: S, store_artifact: Optional[StoreArtifact]) -> T:
        pass


@runtime_checkable
class CallableWithGradientStoringArtifacts(
    CallableStoringArtifacts[np.ndarray, T], Protocol
):
    """A callable with gradient that stored artifacts."""

    def gradient(self, params: np.ndarray) -> np.ndarray:
        pass


def has_store_artifact_param(function) -> bool:
    """Determine if given callable is capable of storing artifacts.

    :param function: a callable to be checked.
    :return: True, if `function` has store_artifact parameter and False otherwise.
    """
    try:
        return "store_artifact" in signature(function, follow_wrapped=True).parameters
    except ValueError:
        # Rationale: the only callables that are of interest to us that aren't supported by
        # signature that I am aware of are numpy ufunc's. Obviously, they don't have
        # store_artifact parameter.
        return False


class FunctionWithGradient(NamedTuple):
    """A callable with gradient."""

    function: Callable[[np.ndarray], np.ndarray]
    gradient: Callable[[np.ndarray], np.ndarray]

    def __call__(self, params: np.ndarray) -> float:
        return self.function(params)


class FunctionWithGradientStoringArtifacts(NamedTuple):
    """A callable with gradient that also stores artifacts."""

    function: CallableStoringArtifacts
    gradient: Callable[[np.ndarray], np.ndarray]

    def __call__(
        self, params: np.ndarray, store_artifact: StoreArtifact = None
    ) -> float:
        return self.function(params, store_artifact)


def function_with_gradient(
    function: Union[Callable[[np.ndarray], float], CallableStoringArtifacts],
    gradient: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Combine function and gradient into an entity adhering to protocols used by history recorder.

    Note that this is a preferred method for adding gradient to your function,
    as it should automatically detect whether the function stores artifact or not.
    """
    if has_store_artifact_param(function):
        return FunctionWithGradientStoringArtifacts(function, gradient)
    else:
        return FunctionWithGradient(function, gradient)
