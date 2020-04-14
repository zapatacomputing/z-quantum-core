from .rest import start_proxy
from ...circuit import save_circuit_template_params, load_circuit_template_params, generate_random_ansatz_params
from ...utils import load_value_estimate, save_value_estimate, ValueEstimate
from multiprocessing import Process
import socket
import json
import http.client
import time
import numpy as np
import subprocess

import unittest

class TestOptimizationServer(unittest.TestCase):

    def setUp(self):
        self.listening_port = 8080
        self.not_listening_port = 8888

        self.proxy_process = Process(target=start_proxy, args=[self.listening_port])
        self.proxy_process.start()

        # Get the proxy IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        self.ipaddress = str(s.getsockname()[0])
        s.close()

        time.sleep(2)

    def test_ping_204(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)
        connection.request('GET', '/')
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

    def test_ping_000(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.not_listening_port), timeout=2)
        self.assertRaises(ConnectionRefusedError, lambda:connection.request('GET', '/'))

    def test_get_starting_status(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)
        connection.request('GET', '/status')
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)
        # Assert that response body is STARTING
        self.assertEqual(response.read().decode("utf-8"), "STARTING")

    def test_post_status(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # Check that status is STARTING
        connection.request('GET', '/status')
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)
        # Assert that response body is STARTING
        self.assertEqual(response.read().decode("utf-8"), "STARTING")
        
        # set status to be OPTIMIZING
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # Check that status is OPTIMIZING
        connection.request('GET', '/status')
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)
        # Assert that response body is OPTIMIZING
        self.assertEqual(response.read().decode("utf-8"), "OPTIMIZING")

    def test_unsuccessful_post_invalid_status(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # Check that status is STARTING
        connection.request('GET', '/status')
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)
        # Assert that response body is STARTING
        self.assertEqual(response.read().decode("utf-8"), "STARTING")
        
        # attempt to set status to be INVALID STATUS
        connection.request('POST', '/status', body="INVALID STATUS")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 400)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('status') != -1)

    def test_post_current_argument_values(self):
        params = np.random.random((2,2))
        save_circuit_template_params(params, 'proxy_test_current_argument_values_artifact.json')
        with open('proxy_test_current_argument_values_artifact.json', 'r') as f:
            data = json.load(f)

        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # set status to be OPTIMIZING
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        connection.request('POST', '/cost-function-argument-values', body=json.JSONEncoder().encode(data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)
        # decode id from response
        id_from_argument_value_post = response.read().decode("utf-8")

        connection.request('GET', '/cost-function-argument-values')
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)

        # remove id from response and verify it is correct
        response_json = json.loads(response.read().decode("utf-8"))
        response_id = response_json.pop("optimization-evaluation-id")
        self.assertEqual(id_from_argument_value_post, response_id)

        # assert argument values are same as above
        with open('proxy_test_current_argument_values_artifact_from_proxy.json', 'w') as f:
            f.write(json.dumps(response_json))
        new_data_loaded_from_file = load_circuit_template_params('proxy_test_current_argument_values_artifact_from_proxy.json')
        np.testing.assert_array_equal(params, new_data_loaded_from_file)

    def test_unsuccessful_post_current_argument_values_wrong_status(self):
        params = np.random.random((2,2))
        save_circuit_template_params(params, 'proxy_test_current_argument_values_artifact.json')
        with open('proxy_test_current_argument_values_artifact.json', 'r') as f:
            data = json.load(f)

        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # set status to be EVALUATING - new argument values should not be able to
        # be posted while that is the status
        connection.request('POST', '/status', body="EVALUATING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        connection.request('POST', '/cost-function-argument-values', body=json.JSONEncoder().encode(data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 409)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('status') != -1)

    def test_unsuccessful_post_argument_values_invalid_JSON(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # set status to be OPTIMIZING
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        connection.request('POST', '/cost-function-argument-values', body="invalid JSON")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 400)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('format') != -1)

    def test_unsuccessful_post_argument_values_no_keys_JSON(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # set status to be OPTIMIZING
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        connection.request('POST', '/cost-function-argument-values', body=json.JSONEncoder().encode("invalid JSON"))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 400)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('dict') != -1)

    def test_unsuccessful_post_argument_values_None(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # set status to be OPTIMIZING
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        connection.request('POST', '/cost-function-argument-values', body=None)
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 400)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('format') != -1)

    def test_post_result(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # POST argument values to allow proxy to verify that id that comes in with
        # result POST are correct
        params = np.random.random((2,2))
        save_circuit_template_params(params, 'proxy_test_current_argument_values_artifact.json')
        with open('proxy_test_current_argument_values_artifact.json', 'r') as f:
            arg_val_data = json.load(f)

        # set status to be OPTIMIZING in order to POST argument values
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # POST argument values
        connection.request('POST', '/cost-function-argument-values', body=json.JSONEncoder().encode(arg_val_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)
        # decode id from response
        id_from_argument_value_post = response.read().decode("utf-8")

        # set status to be EVALUATING
        connection.request('POST', '/status', body="EVALUATING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # make cost function result
        result = ValueEstimate(1.5,10.0)
        save_value_estimate(result, 'proxy_test_results_artifact.json')
        with open('proxy_test_results_artifact.json', 'r') as f:
            result_data = json.load(f)
        result_data["optimization-evaluation-id"] = id_from_argument_value_post
        
        # POST cost function result
        connection.request('POST', '/cost-function-results', body=json.JSONEncoder().encode(result_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # GET cost function result
        connection.request('GET', '/cost-function-results', body=id_from_argument_value_post)
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)

        # remove id from response and verify it is correct
        response_string = response.read().decode("utf-8")
        print(response_string)
        response_json = json.loads(response_string)
        response_id = response_json.pop("optimization-evaluation-id")
        self.assertEqual(id_from_argument_value_post, response_id)

        # assert result is same as above
        with open('proxy_test_results_artifact_from_proxy.json', 'w') as f:
            f.write(json.dumps(response_json))
        new_data_loaded_from_file = load_value_estimate('proxy_test_results_artifact_from_proxy.json')
        self.assertEqual(result.value, new_data_loaded_from_file.value)
        self.assertEqual(result.precision, new_data_loaded_from_file.precision)

    def test_unsuccessful_get_result_no_id(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # POST argument values to allow proxy to verify that argument values that come in with
        # Value POST are correct
        params = np.random.random((2,2))
        save_circuit_template_params(params, 'proxy_test_current_argument_values_artifact.json')
        with open('proxy_test_current_argument_values_artifact.json', 'r') as f:
            arg_val_data = json.load(f)

        # set status to be OPTIMIZING in order to POST argument values
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # POST argument values
        connection.request('POST', '/cost-function-argument-values', body=json.JSONEncoder().encode(arg_val_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)
        # decode id from response
        id_from_argument_value_post = response.read().decode("utf-8")

        # set status to be EVALUATING
        connection.request('POST', '/status', body="EVALUATING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # make cost function result
        result = ValueEstimate(1.5,10.0)
        save_value_estimate(result, 'proxy_test_results_artifact.json')
        with open('proxy_test_results_artifact.json', 'r') as f:
            result_data = json.load(f)
        result_data["optimization-evaluation-id"] = id_from_argument_value_post
        
        # POST cost function result
        connection.request('POST', '/cost-function-results', body=json.JSONEncoder().encode(result_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # GET cost function result
        connection.request('GET', '/cost-function-results') # Will fail bc there's no ID in the body
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 400)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('id') != -1)

    def test_unsuccessful_get_result_wrong_id(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # POST argument values to allow proxy to verify that argument values that come in with
        # Value POST are correct
        params = np.random.random((2,2))
        save_circuit_template_params(params, 'proxy_test_current_argument_values_artifact.json')
        with open('proxy_test_current_argument_values_artifact.json', 'r') as f:
            arg_val_data = json.load(f)

        # set status to be OPTIMIZING in order to POST argument values
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # POST argument values
        connection.request('POST', '/cost-function-argument-values', body=json.JSONEncoder().encode(arg_val_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)
        # decode id from response
        id_from_argument_value_post = response.read().decode("utf-8")

        # set status to be EVALUATING
        connection.request('POST', '/status', body="EVALUATING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # make cost function result
        result = ValueEstimate(1.5,10.0)
        save_value_estimate(result, 'proxy_test_results_artifact.json')
        with open('proxy_test_results_artifact.json', 'r') as f:
            result_data = json.load(f)
        result_data["optimization-evaluation-id"] = id_from_argument_value_post
        
        # POST cost function result
        connection.request('POST', '/cost-function-results', body=json.JSONEncoder().encode(result_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # GET cost function result
        connection.request('GET', '/cost-function-results', body="wrongid")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 400)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('id') != -1)

    def test_unsuccessful_post_result_wrong_id(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # POST argument values to allow proxy to verify that argument values that come in with
        # Value POST are correct
        params = np.random.random((2,2))
        save_circuit_template_params(params, 'proxy_test_current_argument_values_artifact.json')
        with open('proxy_test_current_argument_values_artifact.json', 'r') as f:
            arg_val_data = json.load(f)

        # set status to be OPTIMIZING in order to POST argument values
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # POST argument values
        connection.request('POST', '/cost-function-argument-values', body=json.JSONEncoder().encode(arg_val_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)

        # set status to be EVALUATING
        connection.request('POST', '/status', body="EVALUATING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # make cost function result
        result = ValueEstimate(1.5,10.0)
        save_value_estimate(result, 'proxy_test_results_artifact.json')
        with open('proxy_test_results_artifact.json', 'r') as f:
            result_data = json.load(f)
        result_data["optimization-evaluation-id"] = "wrongID"
        
        # POST cost function result
        connection.request('POST', '/cost-function-results', body=json.JSONEncoder().encode(result_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 409)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('id') != -1)

    def test_unsuccessful_post_result_no_id(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # POST argument values to allow proxy to verify that argument values that come in with
        # Value POST are correct
        params = np.random.random((2,2))
        save_circuit_template_params(params, 'proxy_test_current_argument_values_artifact.json')
        with open('proxy_test_current_argument_values_artifact.json', 'r') as f:
            arg_val_data = json.load(f)

        # set status to be OPTIMIZING in order to POST argument values
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # POST argument values
        connection.request('POST', '/cost-function-argument-values', body=json.JSONEncoder().encode(arg_val_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)

        # set status to be EVALUATING
        connection.request('POST', '/status', body="EVALUATING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # make cost function result
        result = ValueEstimate(1.5,10.0)
        save_value_estimate(result, 'proxy_test_results_artifact.json')
        with open('proxy_test_results_artifact.json', 'r') as f:
            result_data = json.load(f)
        
        # POST cost function result
        connection.request('POST', '/cost-function-results', body=json.JSONEncoder().encode(result_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 400)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('id') != -1)

    def test_unsuccessful_post_result_no_keys_JSON(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # set status to be EVALUATING
        connection.request('POST', '/status', body="EVALUATING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)
        
        # POST cost function result
        connection.request('POST', '/cost-function-results', body=json.JSONEncoder().encode("invalidJSON"))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 400)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('dict') != -1)

    def test_unsuccessful_post_result_body_None(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # POST argument values to allow proxy to verify that argument values that come in with
        # Value POST are correct
        params = np.random.random((2,2))
        save_circuit_template_params(params, 'proxy_test_current_argument_values_artifact.json')
        with open('proxy_test_current_argument_values_artifact.json', 'r') as f:
            arg_val_data = json.load(f)

        # set status to be OPTIMIZING in order to POST argument values
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # POST argument values
        connection.request('POST', '/cost-function-argument-values', body=json.JSONEncoder().encode(arg_val_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)

        # set status to be EVALUATING
        connection.request('POST', '/status', body="EVALUATING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # POST cost function result
        connection.request('POST', '/cost-function-results', body=None)
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 400)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('format') != -1)

    def test_unsuccessful_post_result_wrong_status(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # POST argument values to allow proxy to verify that argument values that come in with
        # Value POST are correct
        params = np.random.random((2,2))
        save_circuit_template_params(params, 'proxy_test_current_argument_values_artifact.json')
        with open('proxy_test_current_argument_values_artifact.json', 'r') as f:
            arg_val_data = json.load(f)

        # set status to be OPTIMIZING in order to POST argument values
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # POST argument values
        connection.request('POST', '/cost-function-argument-values', body=json.JSONEncoder().encode(arg_val_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 200)
        # decode id from response
        id_from_argument_value_post = response.read().decode("utf-8")

        # set status to be EVALUATING
        connection.request('POST', '/status', body="EVALUATING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # make cost function result
        result = ValueEstimate(1.5,10.0)
        save_value_estimate(result, 'proxy_test_results_artifact.json')
        with open('proxy_test_results_artifact.json', 'r') as f:
            result_data = json.load(f)
        result_data["optimization-evaluation-id"] = id_from_argument_value_post
        
        # set status to be OPTIMIZING - new results should not be able to
        # be posted while that is the status
        connection.request('POST', '/status', body="OPTIMIZING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # POST cost function result
        connection.request('POST', '/cost-function-results', body=json.JSONEncoder().encode(result_data))
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 409)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('status') != -1)
        self.assertTrue(response_lower.find('evaluating') != -1)

    def test_unsuccessful_post_result_invalid_JSON(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)

        # set status to be EVALUATING
        connection.request('POST', '/status', body="EVALUATING")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 204)

        # POST cost function result
        connection.request('POST', '/cost-function-results', body="invalid JSON")
        response = connection.getresponse()
        self.assertEqual(response.getcode(), 400)
        response_lower = response.read().decode("utf-8").lower()
        self.assertTrue(response_lower.find('error') != -1)
        self.assertTrue(response_lower.find('format') != -1)

    def tearDown(self):
        connection = http.client.HTTPConnection(self.ipaddress+":"+str(self.listening_port), timeout=2)
        connection.request('POST', '/shutdown')
        response = connection.getresponse()
        print(response.read().decode("utf-8"))
        self.proxy_process.terminate()

        def is_port_in_use(port):
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex((self.ipaddress, port)) == 0

        while(is_port_in_use(self.listening_port)):
            time.sleep(1)
 
    @classmethod
    def tearDownClass(self):
        subprocess.call(["rm", 'proxy_test_current_argument_values_artifact.json',
                         'proxy_test_current_argument_values_artifact_from_proxy.json',
                         'proxy_test_results_artifact.json',
                         'proxy_test_results_artifact_from_proxy.json'])