# z-quantum-core

## What is it?

`z-quantum-core` is a core library of the scientific code for [Orquestra](https://www.zapatacomputing.com/orquestra/) – the platform developed by [Zapata Computing](https://www.zapatacomputing.com) for performing computations on quantum computers.

`z-quantum-core` provides:
- core functionalities required to run other Orquestra modules, such as the `Circuit` class.
- interfaces for implementing other Orquestra modules, such as backends and optimizers.
- useful tools to support the development of workflows and other scientific projects; such as time evolution, sampling from probability distribution, etc.


## Usage

### Workflow
In order to use `z-quantum-core` in your workflow, you need to add it as a `resource` in your Orquestra workflow:

```yaml
resources:
- name: z-quantum-core
  type: git
  parameters:
    url: "git@github.com:zapatacomputing/z-quantum-core.git"
    branch: "master"
```

and then import in the `resources` argument of your `task`:

```yaml
- - name: my-task
    template: template-1
    arguments:
      parameters:
      - param_1: 1
      - resources: [z-quantum-core]
```

Once that is done you can:
- use any template from the `templates/` directory
- use tasks which import `zquantum.core` in the python code (see below)

### Python

Here's an example how to use methods from `z-quantum-core` in a task:

```python
from zquantum.core.circuit import (build_ansatz_circuit,
                                   load_circuit_template,
                                   load_circuit_template_params,
                                   save_circuit)

ansatz = load_circuit_template('ansatz.json');
params = load_circuit_template_params('params.json');
circuit = build_ansatz_circuit(ansatz, params);
save_circuit(circuit, 'circuit.json')
```

Even though it's intended to be used with Orquestra, `z-quantum-core` can be used as a standalone Python module.
This can be done by running `pip install .` from the `src/` directory.

## Development and Contribution

- If you'd like to report a bug/issue please create a new issue in this repository.
- If you'd like to contribute, please create a pull request.

### Running tests

Unit tests for this project can be run using `pytest .` from the main directory.
