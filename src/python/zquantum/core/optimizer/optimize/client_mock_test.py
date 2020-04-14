from .client_mock import MockedClient

import unittest
import json


class TestOptimizationServer(unittest.TestCase):

    def setUp(self):
        pass

    def test_return_zero(self):
        client = MockedClient("fake ip", "fake port")
        self.assertEqual(0, client.cost_func())

        client = MockedClient("fake ip", "fake port", cost_func="return_zero")
        self.assertEqual(0, client.cost_func())

    def test_return_x_squared(self):
        client = MockedClient("fake ip", "fake port", cost_func="return_x_squared", parameters='''{
            "schema": "fake-schema-circuit_template_params",
            "parameters": {"real": [2]}}''')
        self.assertEqual(4, client.cost_func())

    def test_parameters(self):
        client = MockedClient("fake ip", "fake port", parameters='''{
            "schema": "fake-schema-circuit_template_params",
            "parameters": {"real": [1, 2]}}''')
        self.assertEqual(client.parameters, '''{
            "schema": "fake-schema-circuit_template_params",
            "parameters": {"real": [1, 2]}}''')

    def test_get_status(self):
        client = MockedClient("fake ip", "fake port")
        self.assertTrue(client.get_status() in ["EVALUATING", "OPTIMIZING"])

    def test_post_status(self):
        client = MockedClient("fake ip", "fake port")

        try:
            client.post_status("fake status")
        except Exception as e:
            self.fail("post_status() failed unexpectedly.\n{}".format(e))

    def test_get_argument_values(self):
        client = MockedClient("fake ip", "fake port", parameters='''{
            "schema": "fake-schema-circuit_template_params",
            "parameters": {"real": [1, 2]}}''')
        self.assertEqual(client.get_argument_values(), '''{
            "schema": "fake-schema-circuit_template_params",
            "parameters": {"real": [1, 2]}}''')

    def test_post_argument_values(self):
        client = MockedClient("fake ip", "fake port", parameters='''{
            "schema": "fake-schema-circuit_template_params",
            "parameters": {"real": [1, 2]}}''')
        self.assertEqual(client.parameters, '''{
            "schema": "fake-schema-circuit_template_params",
            "parameters": {"real": [1, 2]}}''')

        self.assertEqual("MOCKED-ID", client.post_argument_values('''{
            "schema": "fake-schema-circuit_template_params",
            "parameters": {"real": [3, 4]}}'''))
        self.assertEqual(client.parameters, '''{
            "schema": "fake-schema-circuit_template_params",
            "parameters": {"real": [3, 4]}}''')

    def test_get_evaluation_result(self):
        client = MockedClient("fake ip", "fake port")
        eval_result = json.loads(client.get_evaluation_result("fake id"))
        self.assertEqual(eval_result['value'], 0)

    def test_post_evaluation_result(self):
        client = MockedClient("fake ip", "fake port")

        try:
            client.post_evaluation_result("fake result")
        except Exception as e:
            self.fail("post_evaluation_result() failed unexpectedly.\n{}".format(e))

    def test_start_evaluation(self):
        client = MockedClient("fake ip", "fake port")

        try:
            client.start_evaluation()
        except Exception as e:
            self.fail("start_evaluation() failed unexpectedly.\n{}".format(e))

    def test_finish_evaluation(self):
        client = MockedClient("fake ip", "fake port")

        try:
            client.finish_evaluation("fake result")
        except Exception as e:
            self.fail("finish_evaluation() failed unexpectedly.\n{}".format(e))
        
