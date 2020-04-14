from . import optimize_variational_circuit_with_proxy
from ...interfaces.mock_objects import MockOptimizer
from .client_mock import MockedClient

import http.client
import unittest
import random
import subprocess

class TestOptimizationServer(unittest.TestCase):

    def setUp(self):
        self.port = "1234"
        self.ipaddress = "testing-ip"

    def test_optimize_variational_circuit_with_proxy_all_zero_line(self):
        # Given
        client = MockedClient(self.ipaddress, self.port)
        params = [0, 0]
        optimizer = MockOptimizer()
        # When
        opt_results = optimize_variational_circuit_with_proxy(params,
                        optimizer, client)
        # Then
        self.assertEqual(opt_results['opt_value'], 0)
        self.assertEqual(len(opt_results['opt_params']), 2)
        self.assertEqual(opt_results['history'], [{'optimization-evaluation-ids': ['MOCKED-ID']}])

    def test_optimize_variational_circuit_with_proxy_x_squared(self):
        # Given
        client = MockedClient(self.ipaddress, self.port, "return_x_squared")
        params = [4]
        optimizer = MockOptimizer()
        # When
        opt_results = optimize_variational_circuit_with_proxy(params,
                        optimizer, client)
        # Then
        self.assertGreater(opt_results['opt_value'], 0)
        self.assertEqual(len(opt_results['opt_params']), 1)
        self.assertEqual(opt_results['history'], [{'optimization-evaluation-ids': ['MOCKED-ID']}])

    def test_optimize_variational_circuit_with_proxy_errors(self):
        client = MockedClient(self.ipaddress, self.port)
        params = [0]
        optimizer = MockOptimizer()
        # self.assertRaises(ValueError, lambda: optimize_variational_circuit_with_proxy(
        #     "Not initial params", optimizer, client))

        self.assertRaises(AttributeError, lambda: optimize_variational_circuit_with_proxy(
            params, "Not an optimizer object", "Not a client"))

        self.assertRaises(AttributeError, lambda: optimize_variational_circuit_with_proxy(
            params, optimizer, "Not a client"))
 
    @classmethod
    def tearDownClass(self):
        subprocess.call(["rm", 'client_mock_evaluation_result.json',
                         'current_optimization_params.json'])