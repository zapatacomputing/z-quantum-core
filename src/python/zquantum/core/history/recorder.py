"""Main implementation of recorder."""
from inspect import signature
from typing import TypeVar, Callable, Generic, List, Any, NamedTuple, Dict
from typing_extensions import overload
from ..interfaces.functions import (
    CallableWithGradient,
    CallableStoringArtifacts,
    CallableWithGradientStoringArtifacts,
)
from .save_conditions import always, SaveCondition

T = TypeVar("T")
S = TypeVar("S")


class ArtifactCollection(dict):
    forced: bool = False


class HistoryEntryWithArtifacts(NamedTuple):
    call_number: int
    params: Any
    value: Any
    artifacts: Dict[str, Any]


class HistoryEntry(NamedTuple):
    call_number: int
    params: Any
    value: Any


class SimpleRecorder(Generic[S, T]):
    def __init__(self, target: Callable[[S], T], save_condition: SaveCondition):
        self.predicate = save_condition
        self.target = target
        self.history: List[HistoryEntry] = []
        self.call_number = 0

    def __call__(self, params: S) -> T:
        return_value = self.target(params)
        if self.predicate(return_value, params, self.call_number):
            self.history.append(HistoryEntry(self.call_number, params, return_value))
        self.call_number += 1
        return return_value


class SimpleRecorderWithGradient(SimpleRecorder):
    def __init__(self, target: CallableWithGradient, save_condition: SaveCondition):
        super().__init__(target, save_condition)
        self.gradient = target.gradient


class ArtifactRecorder(Generic[S, T]):
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
                    self.call_number, params, return_value, artifacts
                )
            )
        self.call_number += 1
        return return_value


class ArtifactRecorderWithGradient(ArtifactRecorder):
    def __init__(
        self,
        target: CallableWithGradientStoringArtifacts,
        save_condition: SaveCondition,
    ):
        super().__init__(target, save_condition)
        self.gradient = target.gradient


def store_artifact(artifacts):
    def _store(artifact_name: str, artifact: Any, force: bool = False) -> None:
        artifacts[artifact_name] = artifact
        if force:
            artifacts.forced = True

    return _store


def has_store_artifact_param(function):
    try:
        return "store_artifact" in signature(function, follow_wrapped=True).parameters
    except ValueError:
        # Rationale: the only callables that are of interest to us that aren't supported by
        # signature that I am aware of are numpy ufunc's. Obviously, they don't have
        # store_artifact parameter.
        return False


@overload
def recorder(
    function: CallableWithGradientStoringArtifacts,
    save_condition: SaveCondition = always,
) -> ArtifactRecorderWithGradient:
    pass


@overload
def recorder(
    function: CallableStoringArtifacts[S, T], save_condition: SaveCondition = always
) -> ArtifactRecorder[S, T]:
    pass


@overload
def recorder(
    function: CallableWithGradient, save_condition: SaveCondition = always
) -> SimpleRecorderWithGradient:
    pass


@overload
def recorder(
    function: Callable[[S], T], save_condition: SaveCondition = always
) -> SimpleRecorder[S, T]:
    pass


def recorder(function, save_condition: SaveCondition = always):
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
