import pytest
from zquantum.core.interfaces.backend_test import QuantumBackendTests
from zquantum.core.interfaces.mock_objects import MockQuantumBackend


@pytest.fixture
def backend():
    return MockQuantumBackend()


class TestMockQuantumBackend(QuantumBackendTests):
    pass
