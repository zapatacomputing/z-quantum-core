from unittest.mock import Mock

import pytest
from zquantum.core.wip.compatibility_tools import compatible_with_old_type


class WipType:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return type(self) == type(other) and self.x == other.x

    def __repr__(self):
        return f"WipType({self.x})"


class OldType:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return type(self) == type(other) and self.x == other.x

    def __repr__(self):
        return f"OldType({self.x})"


def translate_old_to_wip(old: OldType):
    return WipType(old.x)


class TestUsesWipTypeDecorator:
    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((1, "test", [5, 7]), {}),
            ((WipType(1),), {}),
            ((WipType(2), 1, "test"), {"x": 10}),
            ((1, 2), {"x": WipType(3), "y": WipType(4)}),
        ],
    )
    def test_uses_wip_type_uses_original_callable_if_no_arguments_of_old_type_are_passed(
        self, args, kwargs
    ):
        original_func = Mock()
        decorated_func = compatible_with_old_type(
            old_type=OldType,
            translate_old_to_wip=translate_old_to_wip,
        )(original_func)

        assert decorated_func(*args, **kwargs) == original_func.return_value
        original_func.assert_called_once_with(*args, **kwargs)

    @pytest.mark.parametrize(
        "original_args, original_kwargs, translated_args, translated_kwargs",
        [
            ((OldType(1),), {}, (WipType(1),), {}),
            ((OldType(2), 1, "test"), {"x": 10}, (WipType(2), 1, "test"), {"x": 10}),
            ((1, 2), {"x": OldType(3)}, (1, 2), {"x": WipType(3)}),
        ],
    )
    def test_uses_wip_type_translates_old_type_to_wip_type(
        self, original_args, original_kwargs, translated_args, translated_kwargs
    ):
        original_func = Mock()
        decorated_func = compatible_with_old_type(
            old_type=OldType,
            translate_old_to_wip=translate_old_to_wip,
        )(original_func)

        assert (
            decorated_func(*original_args, **original_kwargs)
            == original_func.return_value
        )
        original_func.assert_called_once_with(*translated_args, **translated_kwargs)

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((OldType(1),), {}),
            ((OldType(2), 1, "test"), {"x": 10}),
            ((1, 2), {"x": OldType(3)}),
        ],
    )
    def test_deprecation_warning_is_raised_if_deprecation_msg_is_not_none_and_old_type_is_passed(
        self, args, kwargs
    ):
        original_func = Mock()
        deprecation_msg = "OldType will soon be deprecated"
        decorated_func = compatible_with_old_type(
            old_type=OldType,
            translate_old_to_wip=translate_old_to_wip,
            deprecation_msg=deprecation_msg,
        )(original_func)

        with pytest.deprecated_call():
            decorated_func(*args, **kwargs)

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((WipType(1),), {}),
            ((WipType(2), 1, "test"), {"x": 10}),
            ((1, 2), {"x": WipType(3)}),
        ],
    )
    def test_deprecation_warning_is_not_raised_if_deprecation_msg_is_none(
        self, args, kwargs
    ):
        original_func = Mock()
        deprecation_msg = "OldType will soon be deprecated"
        decorated_func = compatible_with_old_type(
            old_type=OldType,
            translate_old_to_wip=translate_old_to_wip,
            deprecation_msg=deprecation_msg,
        )(original_func)

        with pytest.warns(None) as warnings_record:
            decorated_func(*args, **kwargs)

        assert not warnings_record

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((OldType(1),), {}),
            ((OldType(2), 1, "test"), {"x": 10}),
            ((1, 2), {"x": OldType(3)}),
        ],
    )
    def test_fallback_function_is_used_if_translation_fails(self, args, kwargs):
        original_func = Mock()
        fallback_func = Mock()

        def _malfunctioning_translation_func(obj):
            raise ValueError()

        decorated_func = compatible_with_old_type(
            old_type=OldType,
            translate_old_to_wip=_malfunctioning_translation_func,
            fallback_function=fallback_func,
        )(original_func)

        assert decorated_func(*args, **kwargs) == fallback_func.return_value
        original_func.assert_not_called()
        fallback_func.assert_called_once_with(*args, **kwargs)

    @pytest.mark.parametrize(
        "original_args, original_kwargs, translated_args, translated_kwargs",
        [
            (((OldType(1), 2, OldType(2)),), {}, ((WipType(1), 2, WipType(2)),), {}),
            ((), {"x": [OldType(0), OldType(4)]}, (), {"x": [WipType(0), WipType(4)]}),
            (
                ([OldType(1), "test"],),
                {"x": (OldType(2), 3)},
                ([WipType(1), "test"],),
                {"x": (WipType(2), 3)},
            ),
        ],
    )
    def test_old_type_objects_are_translated_if_they_occur_in_considered_iterables(
        self, original_args, original_kwargs, translated_args, translated_kwargs
    ):
        original_func = Mock()
        decorated_func = compatible_with_old_type(
            old_type=OldType,
            translate_old_to_wip=translate_old_to_wip,
            consider_iterable_types=(list, tuple),
        )(original_func)

        assert (
            decorated_func(*original_args, **original_kwargs)
            == original_func.return_value
        )
        original_func.assert_called_once_with(*translated_args, **translated_kwargs)

    @pytest.mark.parametrize(
        "args, kwargs, iterable_types",
        [
            (((OldType(1), 2, OldType(2)),), {}, [list]),
            ((), {"x": [OldType(0), OldType(4)]}, [tuple]),
            (([OldType(1), "test"],), {"x": (OldType(2), 3)}, []),
        ],
    )
    def test_old_type_objects_are_not_translated_if_they_occur_in_not_considered_iterables(
        self, args, kwargs, iterable_types
    ):
        original_func = Mock()
        decorated_func = compatible_with_old_type(
            old_type=OldType,
            translate_old_to_wip=translate_old_to_wip,
            consider_iterable_types=iterable_types,
        )(original_func)

        assert decorated_func(*args, **kwargs) == original_func.return_value
        original_func.assert_called_once_with(*args, **kwargs)
