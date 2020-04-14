from flask import Flask, escape, request
from werkzeug.wrappers import Response
import time
import subprocess
import socket
import json
import uuid

# Current status of optimization.
#   If "STARTING", webserver has been started, but optimizer has not yet initalized.
#   If "OPTIMIZING", no argument values are ready for evaluation/waiting for optimizer.
#   If "EVALUATING", at least one set of argument values is ready for evaluation/waiting for cost function evaluation.
OPTIMIZER_STATUS = "STARTING"

# Allowed status messages for the optimizer.
ALLOWED_STATUSES = ["OPTIMIZING", "EVALUATING", "DONE"]

CURRENT_ARGUMENT_VALUES = None

CURRENT_ID = None

CURRENT_RESULTS = None

app = Flask(__name__)

@app.route('/')
def ping(methods = ['GET']):
    ''' Route Handler for ping
    ARGS: None
    RETURNS: HTTP Response
    '''
    response = Response()
    response.status_code = 204
    return response

@app.route('/status', methods = ['GET', 'POST'])
def status():
    ''' Route Handler for GET and POST methods of optimizer status.

    ARGS: None
    RETURNS: HTTP Response
    '''
    global OPTIMIZER_STATUS, ALLOWED_STATUSES

    if request.method == "GET":
        response = Response(OPTIMIZER_STATUS)
        response.status_code = 200
        return response

    elif request.method == "POST":
        # Get status of optimizer from request
        request_status = request.data.decode("utf-8")

        # ensure new status is one of allowed statuses
        if request_status not in ALLOWED_STATUSES:
            response = Response("ERROR: Status: {} is invalid. Refer to documentation".format(request_status))
            response.status_code = 400
            return response
        
        response = Response()
        response.status_code = 204
        OPTIMIZER_STATUS = request_status
        return response

@app.route('/cost-function-argument-values', methods = ['GET', 'POST'])
def cost_function_argument_values():
    ''' Route Handler for GET and POST methods of current argument value evaluation for optimizer.

    ARGS: None
    RETURNS: HTTP Response
    '''
    global OPTIMIZER_STATUS, CURRENT_ARGUMENT_VALUES, CURRENT_ID

    if request.method == "GET":
        response = Response(json.JSONEncoder().encode(CURRENT_ARGUMENT_VALUES))
        response.status_code = 200
        return response

    elif request.method == "POST":
        if OPTIMIZER_STATUS != "OPTIMIZING":
            response = Response("ERROR: Cannot POST argument values while waiting for evaluation of previous argument values. Must wait until status is \"OPTIMIZING\".")
            response.status_code = 409
            return response

        # Get argument values of optimizer from request
        request_body = request.data.decode("utf-8")

        # Check that argument values are in JSON format and store if so
        try:
            argument_values = json.loads(request_body)
            argument_values.keys()
        except ValueError:
            response = Response("ERROR: Argument values: {} are invalid. Argument values must be in valid JSON format.".format(request_body))
            response.status_code = 400
            return response
        except TypeError:
            response = Response("ERROR: Argument values: {} are invalid. Argument values must have a valid JSON type.".format(request_body))
            response.status_code = 400
            return response
        except AttributeError:
            response = Response("ERROR: Results: {} are invalid. Argument values must be a dict.".format(request_body))
            response.status_code = 400
            return response

        CURRENT_ARGUMENT_VALUES = argument_values

        # generate and store uuid
        CURRENT_ID = uuid.uuid4().hex

        # store current argument values with ID
        CURRENT_ARGUMENT_VALUES["optimization-evaluation-id"] = CURRENT_ID

        # encode ID in response
        response = Response(CURRENT_ID)
        response.status_code = 200
        return response

@app.route('/cost-function-results', methods = ['GET', 'POST'])
def cost_function_results():
    ''' Route Handler for GET and POST methods of current result value for optimizer.

    ARGS: None
    RETURNS: HTTP Response
    '''
    global OPTIMIZER_STATUS, CURRENT_RESULTS, CURRENT_ID

    if request.method == "GET":
        # decode request for id
        requested_id = request.data.decode("utf-8")

        # check to make sure requested id value is current id
        if requested_id != CURRENT_ID:
            response = Response("ERROR: ID: {} is invalid. Requested ID must be current ID".format(requested_id))
            response.status_code = 400
            return response

        response = Response(json.JSONEncoder().encode(CURRENT_RESULTS))
        response.status_code = 200
        return response

    elif request.method == "POST":
        if OPTIMIZER_STATUS != "EVALUATING":
            response = Response("ERROR: Cannot POST result while waiting for new argument values. Must wait until status is \"EVALUATING\".")
            response.status_code = 409
            return response
        
        # Get results from request
        request_body = request.data.decode("utf-8")

        # Check that body is in JSON format
        try:
            results = json.loads(request_body)
            results.keys()
        except ValueError:
            response = Response("ERROR: Results: {} are invalid. Results must have a valid JSON format.".format(request_body))
            response.status_code = 400
            return response
        except TypeError:
            response = Response("ERROR: Results: {} are invalid. Results must have a valid JSON type.".format(request_body))
            response.status_code = 400
            return response
        except AttributeError:
            response = Response("ERROR: Results: {} are invalid. Results must be a dict.".format(request_body))
            response.status_code = 400
            return response

        # Check to make sure optimization-evaluation-id exists
        if "optimization-evaluation-id" not in results.keys():
            # bad request error
            response = Response("ERROR: Results: {} is invalid. Results must contain optimization-evaluation-id field for verification.".format(request_body))
            response.status_code = 400
            return response
        optimization_evaluation_id = results["optimization-evaluation-id"]
        
        if optimization_evaluation_id != CURRENT_ID:
            # conflict error, ID in request is not current ID of optimization
            response = Response("ERROR: ID: {} is invalid. Does not match ID: {} of argument values currently waiting for evaluation.".format(optimization_evaluation_id, CURRENT_ID))
            response.status_code = 409
            return response

        CURRENT_RESULTS = results

        response = Response()
        response.status_code = 204
        return response

@app.route('/shutdown', methods=['POST'])
def shutdown():
    ''' Route Handler POST method to shut down proxy

    ARGS: None
    RETURNS: 
        string indicating proxy is shutting down as http Response
    '''
    shutdown_proxy(request.environ)
    return 'Server shutting down...'

def start_proxy(port):
    ''' Start proxy and listen on port

    ARGS: 
        port (string): port on which to listen

    RETURNS:
        No returns
    '''
    # Get the current IP address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ipaddress = s.getsockname()[0]
    s.close()

    # Run webserver
    subprocess.call(["flask", "run", "--host="+ipaddress, "-p", str(port)])

def is_proxy_running(port):
    ''' Check if proxy is listening on port

    ARGS: 
        port (string): port on which to listen

    RETURNS:
        Boolean (True if proxy is running)
    '''
    # Get the current IP address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ipaddress = str(s.getsockname()[0])
    s.close()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((ipaddress, port)) == 0

def shutdown_proxy(environ):
    ''' Shut down the proxy gracefully

    ARGS: 
        environ (dict): Request environment

    RETURNS:
        No returns
    '''
    if not 'werkzeug.server.shutdown' in environ:
        raise RuntimeError('Not running the development server')
    environ['werkzeug.server.shutdown']()