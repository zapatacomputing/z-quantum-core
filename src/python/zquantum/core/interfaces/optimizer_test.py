import unittest
import numpy as np

from ..cost_function import BasicCostFunction

def rosen(x):
     """The Rosenbrock function"""
     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def simple(x):
     return sum(x**2.0)

class OptimizerTests(object):
    # To run tests with this base class, the following variables need to be properly initialized in the child class:
        # self.optimizers

    def test_optimization(self):
        for optimizer in self.optimizers:
            cost_function = BasicCostFunction(rosen)
            results = optimizer.minimize(cost_function, initial_params=[0, 0])
            self.assertAlmostEqual(results.opt_value, 0, places=5)
            self.assertAlmostEqual(results.opt_params[0], 1, places=4)
            self.assertAlmostEqual(results.opt_params[1], 1, places=4)

            self.assertIn("nfev", results.keys())
            self.assertIn("nit", results.keys())
            self.assertIn("opt_value", results.keys())
            self.assertIn("opt_params", results.keys())
            self.assertIn("history", results.keys())

    def test_optimization_simple_function(self):
        for optimizer in self.optimizers:
            cost_function = BasicCostFunction(simple)
            results = optimizer.minimize(cost_function, initial_params=[1, -1])
            self.assertAlmostEqual(results.opt_value, 0, places=5)
            self.assertAlmostEqual(results.opt_params[0], 0, places=4)
            self.assertAlmostEqual(results.opt_params[1], 0, places=4)

            self.assertIn("nfev", results.keys())
            self.assertIn("nit", results.keys())
            self.assertIn("opt_value", results.keys())
            self.assertIn("opt_params", results.keys())
            self.assertIn("history", results.keys())

