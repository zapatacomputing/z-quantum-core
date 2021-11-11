import pytest


def pytest_collection_modifyitems(config, items):
    """Mark every test file in this folder as unit"""
    for item in items:
        mark = getattr(pytest.mark, "unit")
        if item.name.endswith("_test"):
            item.add_marker(mark)
