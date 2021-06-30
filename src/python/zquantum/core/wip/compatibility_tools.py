"""Tools for building compatibility layers."""
import warnings
from typing import Callable, Optional


class TranslationFailed(Exception):
    """Exception raised if translation failed.

    Note that this class is needed so that we can distinguish between exceptions
    occurring because of translation failure (that should be intercepted) and others
    (which should be propagated).
    """


def _translate_if_needed(
    obj, old_type, translate_old_to_wip, deprecation_msg=None, considered_iterables=None
):
    considered_iterables = [] if not considered_iterables else considered_iterables
    if isinstance(obj, old_type):
        try:
            if deprecation_msg:
                warnings.warn(deprecation_msg, DeprecationWarning)
            return translate_old_to_wip(obj)
        except Exception:
            raise TranslationFailed(
                f"Translation of {obj} failed. Please report this as a bug and provide "
                f"minimal example that triggers this error."
            )
    elif type(obj) in considered_iterables:
        # Note that here we use exact type comparison. This makes for a simpler
        # implementation and it is rather unexpected to see subclasses of list or tuple.
        return type(obj)(
            _translate_if_needed(
                o, old_type, translate_old_to_wip, deprecation_msg, considered_iterables
            )
            for o in obj
        )
    return obj


def compatible_with_old_type(
    old_type,
    translate_old_to_wip,
    deprecation_msg: Optional[str] = None,
    fallback_function=None,
    consider_iterable_types=None,
):
    """Create a decorator marking function as compatible as some "old" type.

    Args:
        old_type: type that the function is compatible with (e.g. old style Circuit)
        translate_old_to_wip: callable mapping old style object to new one.
            It should take a single argument and return a single value.
        deprecation_msg: if and old style object is encountered, a deprecation warning
            is issued with this message. If not provided, not DeprecationWarning is
            ever produced.
        fallback_function: function equivalent to the decorated one, that should be
            used in case of any translation failure. If provided, translation failures
            trigger warning, otherwise they result in an error.
        consider_iterable_types: an iterable of iterable types (e.g. list, tuple)
            that should be also checked for old-style objects. This is supposed to
            work with built-in sequential types.
    Returns:
        A decorator that makes given function using wip objects compatible with
            old-style objects.
    Raises:
        TranslationFailure: if translation failed and no fallback_function is provided.
    """

    def _compatible_with_old_type(wrapped: Callable):
        def _inner(*args, **kwargs):
            try:
                new_args = tuple(
                    _translate_if_needed(
                        obj,
                        old_type,
                        translate_old_to_wip,
                        deprecation_msg,
                        consider_iterable_types,
                    )
                    for obj in args
                )
                new_kwargs = {
                    key: _translate_if_needed(
                        obj,
                        old_type,
                        translate_old_to_wip,
                        deprecation_msg,
                        consider_iterable_types,
                    )
                    for key, obj in kwargs.items()
                }
                return wrapped(*new_args, **new_kwargs)
            except TranslationFailed as err:
                if fallback_function:
                    warnings.warn(
                        f"Translation from {old_type} failed, and fallback function "
                        "will be used. Please report this as a bug and provide minimal "
                        "example that leads to this error."
                    )
                    return fallback_function(*args, **kwargs)
                raise err

        return _inner

    return _compatible_with_old_type
