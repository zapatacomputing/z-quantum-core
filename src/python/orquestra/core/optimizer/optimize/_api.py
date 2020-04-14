from ...circuit import save_circuit_template_params
from ...utils import load_value_estimate, convert_array_to_dict, SCHEMA_VERSION

import scipy
import time
import json
import io

# ---------- Optimize Variational Circuit ----------
def optimize_variational_circuit_with_proxy(initial_params, optimizer, client, **kwargs):
    """Optimizes a variational circuit using proxy architecture.
    
    Arguments:
        initial_params (numpy.ndarray): initial guess for the ansatz parameters.
        method (string): scipy method for optimization
        client (orquestra.core.optimizer.proxy.Client): a client for interacting with
            the proxy

        *** OPTIONAL ***
        keep_value_history (bool): If true, an evaluation is done after every
            iteration of the optimizer and the value is saved
        layers_to_optimize (str): which layers of the ansatz to optimize. Options
            are 'all' and 'last'.
        options (dict): options for scipy optimizer
        **kwargs: keyword arguments passed to orquestra.core.optimization.minimize

    Returns:
        tuple: two-element tuple containing

        - **results** (**dict**): a dictionary with keys `value`, `status`,
                `success`, `nfev`, and `nit`
        - **optimized_params** (**numpy.array**): the optimized parameters
    """
    current_iteration = 0
    keep_value_history = optimizer.options['keep_value_history']

    # Optimization Results Object

    opt_results = scipy.optimize.OptimizeResult({'value': None, 'status': None, 'success': False, 'nfev': 0,
        'nit': 0, 'history':[{'optimization-evaluation-ids': []}]})
    history =  [{'optimization-evaluation-ids': []}]
    # Define cost function that interacts with client
    def cost_function(params):
        nonlocal history, current_iteration
        # Encode params to json string
        save_circuit_template_params(params, 'current_optimization_params.json')
        with open('current_optimization_params.json', 'r') as f:
            current_params_string = f.read()

        # POST params to proxy
        evaluation_id = client.post_argument_values(current_params_string)

        # SAVE ID to optimization result['history']
        history[current_iteration]['optimization-evaluation-ids'].append(evaluation_id)

        # POST status to EVALUATING
        client.post_status("EVALUATING")

        # WAIT for status to be OPTIMIZING
        while client.get_status() != "OPTIMIZING":
            time.sleep(1)

        # GET cost function evaluation from proxy
        evaluation_string = client.get_evaluation_result(evaluation_id)
        value_estimate = load_value_estimate(io.StringIO(evaluation_string))

        return value_estimate.value

    # Define callback function called after each iteration of the optimizer
    def callback(params):
        nonlocal history, current_iteration
        history[current_iteration]['params'] = params

        print("\nFinsished Iteration: {}".format(current_iteration), flush=True)
        print("Current Parameters: {}".format(params), flush=True)

        # If getting the value history, perform an evaluation with current params
        if keep_value_history:
            history[current_iteration]['value'] = cost_function(params)

            print("Current Value: {}".format(
                history[current_iteration]['value']), flush=True)
        
        print("Starting Next Iteration...", flush=True)

        # Update currrent_iteration index and add new blank history
        current_iteration += 1
        history.append({'optimization-evaluation-ids': []})
    
    # POST status to OPTIMIZING
    client.post_status("OPTIMIZING")
    
    # Perform the minimization
    opt_results = optimizer.minimize(cost_function,
                                     initial_params,
                                     callback=callback)
    
    # Update opt_results object
    opt_results['history'] = history

    # Since a new history element is added in the callback function, if there is 
    #   at least one iteration, there is an empty history element at the end
    #   that must be removed
    if 'nit' in opt_results.keys() and opt_results.nit > 0:
        del opt_results['history'][-1]

    # POST status to DONE
    client.post_status("DONE")

    return opt_results