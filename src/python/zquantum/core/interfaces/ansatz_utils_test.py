"""Test cases for ansatz-related utilities."""
import unittest
from .ansatz_utils import DynamicProperty, ansatz_property


class DynamicPropertyTests(unittest.TestCase):

    def test_uses_default_value(self):
        """DynamicProperty should use default value if it wasn't overwritten."""
        class MyCls:
            x = DynamicProperty(name="x", default_value=-15)

        obj = MyCls()
        self.assertEqual(obj.x, -15)

    def test_can_be_set_in_init(self):
        """It should be possible to initialize DynamicProperty's value in init method."""
        class MyCls:
            length = DynamicProperty(name="length")

            def __init__(self, length):
                self.length = length

        obj = MyCls(0.5)
        self.assertEqual(obj.length, 0.5)

    def test_stores_values_in_instance(self):
        """Values of DynamicProperty should be instance-dependent (i.e. should have its own copy)."""
        class MyCls:
            height = DynamicProperty(name="height")

        obj1 = MyCls()
        obj2 = MyCls()

        obj1.height = 15
        obj2.height = 30

        self.assertEqual(obj1.height, 15)
        self.assertEqual(obj2.height, 30)


class TestAnsatzProperty(unittest.TestCase):
    """Test cases for ansatz_property.

    Note that we don't relaly need an ansatz intance, we only need to check that _parametrized_circuit is
    set to None.
    """

    class PseudoAnsatz:
        n_layers = ansatz_property(name="n_layers")

        def __init__(self, n_layers):
            self.n_layers = n_layers
            self._parametrized_circuit = None

        @property
        def parametrized_circuit(self):
            if self._parametrized_circuit is None:
                self._parametrized_circuit = f"Circuit with {self.n_layers} layers"
            return self._parametrized_circuit

    def test_setter_resets_parametrized_circuit(self):
        ansatz = self.PseudoAnsatz(n_layers=10)

        # Trigger initial computation of parametrized circuit
        self.assertEqual(ansatz.parametrized_circuit, "Circuit with 10 layers")

        # Change n_layers -> check if it recalculated.
        ansatz.n_layers = 20
        self.assertIsNone(ansatz._parametrized_circuit)
        self.assertEqual(ansatz.parametrized_circuit, "Circuit with 20 layers")
