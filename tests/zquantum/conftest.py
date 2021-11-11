import pytest


def pytest_collection_modifyitems(config, items):
    """Mark every test file in this folder as unit"""
    for item in items:
        mark_name = "unit"
        if mark_name:
            mark = getattr(pytest.mark, mark_name)
            item.add_marker(mark)
