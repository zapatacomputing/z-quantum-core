import http.client
import time
import json


class Client:

    def __init__(self, ip, port):
        ''' Initializes the client object, giving it an http connection to the
        proxy.

        ARGS:
            ip (string): the ip address of the proxy server
            port (string): the port that the proxy is listening on
        '''
        self.connection = http.client.HTTPConnection(ip+":"+port, timeout=2)

    def get_status(self):
        ''' Get the status of the optimization

        ARGS:
            None
        RETURNS:
            The status of the optimization as a string
        '''
        # GET status
        self.connection.request('GET', '/status')
        response = self.connection.getresponse()

        if response.getcode() != 200:
            raise RuntimeError("Exiting with response: Status code {}: {}".format(response.getcode(),response.read().decode("utf-8")))

        return response.read().decode("utf-8")


    def post_status(self, status):
        ''' Post the status of the optimization

        ARGS:
            status (string): The new status

        RETURNS:
            No return
        '''
        # POST status
        self.connection.request('POST', '/status', body=status)
        response = self.connection.getresponse()

        if response.getcode() != 204:
            raise RuntimeError("Exiting with response: Status code {}: {}".format(response.getcode(),response.read().decode("utf-8")))


    def get_argument_values(self):
        ''' Get the argument values

        ARGS:
            None
        RETURNS:
            The argument values as a string in JSON format
        '''
        # GET argument values
        self.connection.request('GET', '/cost-function-argument-values')
        response = self.connection.getresponse()

        if response.getcode() != 200:
            raise RuntimeError("Exiting with response: Status code {}: {}".format(response.getcode(),response.read().decode("utf-8")))

        return response.read().decode("utf-8")


    def post_argument_values(self, argument_values):
        ''' Post the argument values

        ARGS:
            argument values (string): The argument values as a string in JSON format

        RETURNS:
            id (string): id associated with the evaluation of the argument values
        '''
        # POST argument values
        self.connection.request('POST', '/cost-function-argument-values', body=argument_values)
        response = self.connection.getresponse()

        if response.getcode() != 200:
            raise RuntimeError("Exiting with response: Status code {}: {}".format(response.getcode(),response.read().decode("utf-8")))

        # Returns id from response
        return response.read().decode("utf-8")


    def get_evaluation_result(self, id):
        ''' Get the evaluation result

        ARGS:
            id (string): id associated with the evaluation

        RETURNS:
            The evaluation result as a string in JSON format
        '''
        # GET evaluation result
        self.connection.request('GET', '/cost-function-results', body=id)
        response = self.connection.getresponse()

        if response.getcode() != 200:
            raise RuntimeError("Exiting with response: Status code {}: {}".format(response.getcode(),response.read().decode("utf-8")))
        
        return response.read().decode("utf-8")


    def post_evaluation_result(self, evaluation_result):
        ''' Post the evaluation result

        ARGS:
            evaluation_result (string): The evaluation result as a string in JSON format

        RETURNS:
            No return
        '''
        # POST evaluation result with argument values
        self.connection.request('POST', '/cost-function-results', body=evaluation_result)
        response = self.connection.getresponse()

        if response.getcode() != 204:
            raise RuntimeError("Exiting with response: Status code {}: {}".format(response.getcode(),response.read().decode("utf-8")))


    def start_evaluation(self):
        ''' Get the argument values and id from the proxy when ready for evaluation

        ARGS:
            None

        RETURNS:
            The argument values as a string in JSON format
            The id associated with the argument values in string format 
        '''
        status = self.get_status()

        # GET status while status != "EVALUATING"
        while status != "EVALUATING" :
            status = self.get_status()
            time.sleep(1)

        # GET argument values
        argument_values_string = self.get_argument_values()

        try:
            argument_values_json = json.loads(argument_values_string)
        except Exception as e:
            raise RuntimeError("{}".format(e))

        try:
            id = argument_values_json['optimization-evaluation-id']
        except Exception as e:
            raise RuntimeError("{}".format(e))
        
        return argument_values_string, id


    def finish_evaluation(self, evaluation_result):
        ''' Post the evaluation result to proxy and change status to indicate it is ready for proxy

        ARGS:
            evaluation_result (string): The evaluation result as a string in JSON format

        RETURNS:
            No return
        '''
        self.post_evaluation_result(evaluation_result)
                
        self.post_status("OPTIMIZING")

