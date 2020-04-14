from ...utils import ValueEstimate, save_value_estimate
from ...circuit import load_circuit_template_params

import random
import json
import io

class MockedClient:

    def return_zero(self):
        return 0

    def return_x_squared(self):
        return load_circuit_template_params(io.StringIO(self.parameters))[0]**2


    def __init__(self, ip, port, cost_func="return_zero", parameters=None):
        self.connection = "mocked connection"

        if cost_func == "return_zero":
            self.cost_func = self.return_zero
        elif cost_func == "return_x_squared":
            self.cost_func = self.return_x_squared
        else:
            raise RuntimeError("Cost Function {} not implemented".format(cost_func))

        self.parameters = parameters

    def get_status(self):
        return random.choice(["EVALUATING", "OPTIMIZING"])


    def post_status(self, status):
        pass


    def get_argument_values(self):
        return self.parameters


    def post_argument_values(self, argument_values):
        self.parameters = argument_values
        return "MOCKED-ID"


    def get_evaluation_result(self, id):
        # make cost function result
        result = ValueEstimate(self.cost_func())
        save_value_estimate(result, 'client_mock_evaluation_result.json')
        with open('client_mock_evaluation_result.json', 'r') as f:
            result_data = json.load(f)
        result_data["optimization-evaluation-id"] = "MOCKED-ID"
        return json.JSONEncoder().encode(result_data)


    def post_evaluation_result(self, evaluation_result):
        pass


    # Not Implemented
    def start_evaluation(self):
        pass


    # Not Implemented
    def finish_evaluation(self, evaluation_result):
        pass
