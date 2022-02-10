"""Main implementation of the recorder."""
import copy
from typing import Any, Callable, Dict, Generic, List, NamedTuple, TypeVar

from typing_extensions import overload

from ..interfaces.functions import (
    CallableStoringArtifacts,
    CallableWithGradient,
    CallableWithGradientStoringArtifacts,
    StoreArtifact,
    has_store_artifact_param,
)
from .save_conditions import SaveCondition, always

T = TypeVar("T")
S = TypeVar("S")

NATIVE_RECORDER_ATTRIBUTES = (
    "predicate",
    "history",
    "target",
    "call_number",
    "gradient",
)


class ArtifactCollection(dict):
    """A dict with additional `forced` attribute.

    The `forced` flag is set whenever an artifact is forced into the dictionary
    despite current save_condition being false.
    """

    forced: bool = False


class HistoryEntryWithArtifacts(NamedTuple):
    """A history entry enhanced with artifacts."""

    call_number: int
    params: Any
    value: Any
    artifacts: Dict[str, Any]


class HistoryEntry(NamedTuple):
    """A history entry storing call number, parameters and target function value."""

    call_number: int
    params: Any
    value: Any


def copy_recorder(recorder_to_copy):
    attributes_dict = {
        "target": recorder_to_copy.target,
        "save_condition": recorder_to_copy.predicate,
    }

    recorder_copy = type(recorder_to_copy)(**attributes_dict)
    recorder_copy.call_number = recorder_to_copy.call_number
    recorder_copy.history = recorder_to_copy.history
    return recorder_copy


def deepcopy_recorder(recorder_to_copy, memo):
    attributes_dict = {
        "target": copy.deepcopy(recorder_to_copy.target, memo=memo),
        "save_condition": copy.deepcopy(recorder_to_copy.predicate, memo=memo),
    }

    recorder_copy = type(recorder_to_copy)(**attributes_dict)
    recorder_copy.call_number = recorder_to_copy.call_number
    recorder_copy.history = copy.deepcopy(recorder_to_copy.history, memo=memo)
    return recorder_copy


class SimpleRecorder(Generic[S, T]):
    """A basic recorder that stores history entries.

    Args:
        target: a target function. Calls to the recorder will be propagated to this
          function.
        save_condition: a function determining whether given call should be saved
          to the history. See respective protocol for explanation of this parameter.
    """

    def __init__(self, target: Callable[[S], T], save_condition: SaveCondition):
        self.predicate = save_condition
        self.target = target
        self.history: List[HistoryEntry] = []
        self.call_number = 0

    def __call__(self, params: S) -> T:
        """Call the underlying target function, possibly saving call to the history.

        Args:
            params: argument to be passed to the target function.

        Returns:
            The value returned by the target function.
        """
        return_value = self.target(params)
        if self.predicate(return_value, params, self.call_number):
            self.history.append(
                HistoryEntry(self.call_number, copy.copy(params), return_value)
            )
        self.call_number += 1
        return return_value

    def __getattr__(self, item):
        return getattr(self.target, item)

    def __setattr__(self, key, value):
        if key in NATIVE_RECORDER_ATTRIBUTES:
            return object.__setattr__(self, key, value)
        return setattr(self.target, key, value)

    __copy__ = copy_recorder

    __deepcopy__ = deepcopy_recorder


class SimpleRecorderWithGradient(SimpleRecorder):
    """A recorder saving history entries that works with callables with gradient.

    Except having `gradient` attribute, this recorder is the same as `SimpleRecorder`.
    """

    def __init__(self, target: CallableWithGradient, save_condition: SaveCondition):
        super().__init__(target, save_condition)
        self.gradient = recorder(target.gradient, save_condition)


class ArtifactRecorder(Generic[S, T]):
    """A recorder saving history entries with artifacts.

    Parameters to initializer are the same as for `SimpleRecorder`,
    except the target function should now be capable of storing artifacts.
    """

    def __init__(
        self, target: CallableStoringArtifacts[S, T], save_condition: SaveCondition
    ):
        self.predicate = save_condition
        self.target = target
        self.history: List[HistoryEntryWithArtifacts] = []
        self.call_number = 0

    def __call__(self, params: S) -> T:
        artifacts = ArtifactCollection()
        return_value = self.target(params, store_artifact=store_artifact(artifacts))

        if self.predicate(return_value, params, self.call_number) or artifacts.forced:
            self.history.append(
                HistoryEntryWithArtifacts(
                    self.call_number, copy.copy(params), return_value, artifacts
                )
            )
        self.call_number += 1
        return return_value

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __setattr__(self, name, value):
        if name in ("predicate", "history", "target", "call_number", "gradient"):
            return object.__setattr__(self, name, value)
        return setattr(self.target, name, value)

    __copy__ = copy_recorder

    __deepcopy__ = deepcopy_recorder


class ArtifactRecorderWithGradient(ArtifactRecorder):
    """A recorder storing history entries with artifacts supporting callables with
    gradient.
    """

    def __init__(
        self,
        target: CallableWithGradientStoringArtifacts,
        save_condition: SaveCondition,
    ):
        super().__init__(target, save_condition)
        self.gradient = recorder(target.gradient, save_condition)


def store_artifact(artifacts) -> StoreArtifact:
    """Create a function storing artifacts in given artifacts collection.

    Args:
        artifacts: artifact collection.

    Returns:
        A function with signature:
        _store(artifact_name: str, artifact: Any, force: bool = False) -> None:
        This function is intended to be passed to functions that are capable of
        storing artifacts.
    """

    def _store(artifact_name: str, artifact: Any, force: bool = False) -> None:
        artifacts[artifact_name] = artifact
        if force:
            artifacts.forced = True

    return _store


@overload
def recorder(
    function: CallableWithGradientStoringArtifacts,
    save_condition: SaveCondition = always,
) -> ArtifactRecorderWithGradient:
    """The recorder function: variant for artifact-storing callables with gradient."""


@overload
def recorder(
    function: CallableStoringArtifacts[S, T], save_condition: SaveCondition = always
) -> ArtifactRecorder[S, T]:
    """The recorder function: variant for callables with no gradient that store
    artifacts."""


@overload
def recorder(
    function: CallableWithGradient, save_condition: SaveCondition = always
) -> SimpleRecorderWithGradient:
    """The recorder function: variant for callables with gradient that don't store
    artifacts."""


@overload
def recorder(
    function: Callable[[S], T], save_condition: SaveCondition = always
) -> SimpleRecorder[S, T]:
    """The recorder function: variant for callables without gradient that don't store
    artifacts."""


def recorder(function, save_condition: SaveCondition = always):
    """Create a recorder that is suitable for recording calls to given callable.

    Args:
        function: a callable to be recorded.
        save_condition: a condition on which the calls will be saved. See
          `SaveCondition` protocol for explanation of this parameter. By default
          all calls are saved.

    Returns:
        A callable object (the recorder) wrapping the `function`.
        The return type depends on the passed callable. See overloads defined
        above to check for available variants. Here is a summary:
        - recorder is always callable
        - if `function` has gradient, so does the recorder. Calls to gradient
          and calls made by gradient are NOT recorded.
        - if `function` has possibility to store artifacts (i.e. accepts
          `store_artifact` argument, then so does the recorder.
    """
    with_artifacts = has_store_artifact_param(function)
    with_gradient = isinstance(function, CallableWithGradient)

    if with_artifacts and with_gradient:
        return ArtifactRecorderWithGradient(function, save_condition)
    elif with_artifacts:
        return ArtifactRecorder(function, save_condition)
    elif with_gradient:
        return SimpleRecorderWithGradient(function, save_condition)
    else:
        return SimpleRecorder(function, save_condition)
