from .client import Client

from .server_mock import MockServer

import json
import http.client
import time
import random
import unittest
from werkzeug.wrappers import Response

class TestOptimizationServer(unittest.TestCase):

    def setUp(self):
        self.port = 8080
        self.ipaddress = "localhost"

        self.server = MockServer(port=self.port)
        self.server.start()

        self.possible_status_codes = [200, 204, 400, 404, 409, 500]

        self.client = Client(self.ipaddress, str(self.port))

        time.sleep(2)

    def test_get_status(self):
        status = "OPTIMIZING"
        def callback():
            response = Response(status)
            response.status_code = 200
            return response
        self.server.add_callback_response('/status', callback)

        returned_status = self.client.get_status()

        # check that status is what it was set to
        self.assertEqual(status, returned_status)

    def test_get_status_error(self):
        def callback():
            response = Response()
            response.status_code = random.choice(self.possible_status_codes)

            # Failures should occur when status is not 200
            while response.status_code == 200:
                response.status_code = random.choice(self.possible_status_codes)
            return response

        self.server.add_callback_response('/status', callback)

        # check that raises runtime error
        self.assertRaises(RuntimeError, lambda: self.client.get_status())

    def test_post_status(self):
        def callback():
            response = Response()
            response.status_code = 204
            return response

        self.server.add_callback_response('/status', callback, methods=('POST',))

        # Function call should not fail, but has no returns
        try:
            self.client.post_status("OPTIMIZING")
        except Exception as e:
            self.fail("post_status() failed unexpectedly.\n{}".format(e))

    def test_post_status_error(self):
        def callback():
            response = Response()
            response.status_code = random.choice(self.possible_status_codes)

            # Failures should occur when status is not 204
            while response.status_code == 204:
                response.status_code = random.choice(self.possible_status_codes)
            return response

        self.server.add_callback_response('/status', callback, methods=('POST',))

        # check that raises runtime error
        self.assertRaises(RuntimeError, lambda: self.client.post_status("OPTIMIZING"))

    def test_get_argument_values(self):
        argument_values_string = json.JSONEncoder().encode({'argval1': 'hello'})
        def callback():
            response = Response(argument_values_string)
            response.status_code = 200
            return response
        self.server.add_callback_response('/cost-function-argument-values', callback)

        returned_argument_values_string = self.client.get_argument_values()

        # check that argument values what they were set to
        self.assertEqual(returned_argument_values_string, argument_values_string)

    def test_get_argument_values_error(self):
        def callback():
            response = Response()
            response.status_code = random.choice(self.possible_status_codes)

            # Failures should occur when status is not 200
            while response.status_code == 200:
                response.status_code = random.choice(self.possible_status_codes)
            return response
        self.server.add_callback_response('/cost-function-argument-values', callback)

        # check that raises runtime error
        self.assertRaises(RuntimeError, lambda: self.client.get_argument_values())

    def test_post_argument_values(self):
        id = "this is a test id"
        def callback():
            response = Response(id)
            response.status_code = 200
            return response
        self.server.add_callback_response('/cost-function-argument-values', callback, methods=('POST',))

        argument_values_string = json.JSONEncoder().encode({'argval1': 'hello'})

        returned_id = self.client.post_argument_values(argument_values_string)

        # check that id is decoded from response correctly
        self.assertEqual(returned_id, id)

    def test_post_argument_values_error(self):
        def callback():
            response = Response()
            response.status_code = random.choice(self.possible_status_codes)

            # Failures should occur when status is not 200
            while response.status_code == 200:
                response.status_code = random.choice(self.possible_status_codes)
            return response
            
        self.server.add_callback_response('/cost-function-argument-values', callback, methods=('POST',))

        argument_values_string = json.JSONEncoder().encode({'argval1': 'hello'})

        # check that raises runtime error
        self.assertRaises(RuntimeError, lambda: self.client.post_argument_values(argument_values_string))

    def test_get_evaluation_result(self):
        id = "this is a test id"
        evaluation_result_string = json.JSONEncoder().encode({'evres1': 'hello', 'optimization-evaluation-id': id})

        def callback():
            response = Response(evaluation_result_string)
            response.status_code = 200
            return response

        self.server.add_callback_response('/cost-function-results', callback, methods=('GET',))

        returned_evaluation_result_string = self.client.get_evaluation_result(id)

        # check that evaluation result is decoded from response correctly
        self.assertEqual(returned_evaluation_result_string, evaluation_result_string)

    def test_get_evaluation_result_error(self):
        def callback():
            response = Response()
            response.status_code = random.choice(self.possible_status_codes)

            # Failures should occur when status is not 200
            while response.status_code == 200:
                response.status_code = random.choice(self.possible_status_codes)
            return response
            
        self.server.add_callback_response('/cost-function-results', callback, methods=('GET',))

        id = "this is a test id"

        # check that raises runtime error
        self.assertRaises(RuntimeError, lambda: self.client.get_evaluation_result(id))

    def test_post_evaluation_result(self):
        def callback():
            response = Response()
            response.status_code = 204
            return response

        self.server.add_callback_response('/cost-function-results', callback, methods=('POST',))

        id = "this is a test id"
        evaluation_result_string = json.JSONEncoder().encode({'evres1': 'hello', 'optimization-evaluation-id': id})
        
        # Function call should not fail, but has no returns
        try:
            self.client.post_evaluation_result(evaluation_result_string)
        except Exception as e:
            self.fail("post_evaluation_result() failed unexpectedly.\n{}".format(e))

    def test_post_evaluation_result_error(self):
        def callback():
            response = Response()
            response.status_code = random.choice(self.possible_status_codes)

            # Failures should occur when status is not 204
            while response.status_code == 204:
                response.status_code = random.choice(self.possible_status_codes)
            return response
            
        self.server.add_callback_response('/cost-function-results', callback, methods=('POST',))

        id = "this is a test id"
        evaluation_result_string = json.JSONEncoder().encode({'evres1': 'hello', 'optimization-evaluation-id': id})
        
        # check that raises runtime error
        self.assertRaises(RuntimeError, lambda: self.client.post_evaluation_result(evaluation_result_string))

    def test_start_evaluation(self):
        id = "this is a test id"

        def get_status_callback():
            response = Response(random.choice(["OPTMIZING", "EVALUATING"]))
            response.status_code = 200
            return response

        self.server.add_callback_response('/status', get_status_callback)

        argument_values_string = json.JSONEncoder().encode({'argval1': 'hello', 'optimization-evaluation-id': id})
        def get_argument_values_callback():
            response = Response(argument_values_string)
            response.status_code = 200
            return response
        
        self.server.add_callback_response('/cost-function-argument-values', get_argument_values_callback)

        returned_argument_values_string, returned_id = self.client.start_evaluation()
        self.assertEqual(returned_argument_values_string, argument_values_string)
        self.assertEqual(returned_id, id)

    def test_start_evaluation_error_not_json(self):
        def get_status_callback():
            response = Response(random.choice(["OPTMIZING", "EVALUATING"]))
            response.status_code = 200
            return response

        self.server.add_callback_response('/status', get_status_callback)

        def get_argument_values_callback():
            response = Response(123)
            response.status_code = 200
            return response
        
        self.server.add_callback_response('/cost-function-argument-values', get_argument_values_callback)

        self.assertRaises(RuntimeError, lambda: self.client.start_evaluation())

    def test_start_evaluation_error_no_id_field(self):
        def get_status_callback():
            response = Response(random.choice(["OPTMIZING", "EVALUATING"]))
            response.status_code = 200
            return response

        self.server.add_callback_response('/status', get_status_callback)

        argument_values_string = json.JSONEncoder().encode({'argval1': 'hello'})
        def get_argument_values_callback():
            response = Response(argument_values_string)
            response.status_code = 200
            return response
        
        self.server.add_callback_response('/cost-function-argument-values', get_argument_values_callback)

        self.assertRaises(RuntimeError, lambda: self.client.start_evaluation())

    def test_finish_evaluation(self):
        id = "this is a test id"

        def post_status_callback():
            response = Response()
            response.status_code = 204
            return response

        self.server.add_callback_response('/status', post_status_callback, methods=('POST',))

        def post_evaluation_result_callback():
            response = Response()
            response.status_code = 204
            return response

        self.server.add_callback_response('/cost-function-results', post_evaluation_result_callback, methods=('POST',))

        evaluation_result_string = json.JSONEncoder().encode({'evres1': 'hello', 'optimization-evaluation-id': id})

        try:
            self.client.finish_evaluation(evaluation_result_string)
        except Exception as e:
            self.fail("finish_evaluation() failed unexpectedly.\n{}".format(e))

    def tearDown(self):
        self.server.shutdown_server()

        def is_port_in_use(port):
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex((self.ipaddress, port)) == 0

        while(is_port_in_use(self.port)):
            time.sleep(1)
