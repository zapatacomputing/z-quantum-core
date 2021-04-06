# z-quantum-core

## What is it?

`z-quantum-core` is a core library of the scientific code for [Orquestra](https://www.zapatacomputing.com/orquestra/) – the platform developed by [Zapata Computing](https://www.zapatacomputing.com) for performing computations on quantum computers.

`z-quantum-core` provides:

- core functionalities required to run other Orquestra modules, such as the `Circuit` class.
- interfaces for implementing other Orquestra modules, such as backends and optimizers.
- useful tools to support the development of workflows and other scientific projects; such as time evolution, sampling from probability distribution, etc.

## Usage

### Workflow

In order to use `z-quantum-core` in your workflow, you need to add it as an `import` in your Orquestra workflow:

```yaml
imports:
  - name: z-quantum-core
    type: git
    parameters:
      repository: "git@github.com:zapatacomputing/z-quantum-core.git"
      branch: "master"
```

and then add it in the `imports` argument of your `step`:

```yaml
- name: my-step
  config:
    runtime:
      language: python3
      imports: [z-quantum-core]
```

Once that is done you can:

- use any `z-quantum-core` function by specifying its name and path as follows:

```yaml
- name: generate-parameters
  config:
    runtime:
      language: python3
      imports: [z-quantum-core]
      parameters:
        file: z-quantum-core/steps/circuit.py
        function: generate_random_ansatz_params
```

- use tasks which import `zquantum.core` in the python code (see below)

### Python

Here's an example of how to use methods from `z-quantum-core` in a python task:

```python
from zquantum.core.circuit import (
    build_ansatz_circuit,
    load_circuit_template,
    load_circuit_template_params,
    save_circuit,
)

ansatz = load_circuit_template("ansatz.json")
params = load_circuit_template_params("params.json")
circuit = build_ansatz_circuit(ansatz, params)
save_circuit(circuit, "circuit.json")
```

Even though it's intended to be used with Orquestra, `z-quantum-core` can be also used as a standalone Python module.
To install it, you just need to run `pip install -e .` from the main directory. )

## Development and Contribution

To install the development version, run `pip install -e '.[dev]'` from the main directory. (if using MacOS, you will need single quotes around the []. If using windows, or Linux, you might not need the quotes)

We use pre-commit hooks to check for minor errors before developers can commit/push code (See info about pre-commit `here <https://pre-commit.com/>`\_.). A pre-commit hook can automatically format code, check for linting errors, and give quick feedback to the developer.
It also helps keep commits clean, and git diffs minimal. Please install it via:

`pre-commit install`

Pre-commit will prevent pushes to remote if certain formatting/check do not pass. When pushing, pre-commit will run pytest.
Note that if needed, you can skip these checks with the `--no-verify` option, i.e. `git push --no-verify`.

- If you'd like to report a bug/issue please create a new issue in this repository.
- If you'd like to contribute, please create a pull request.

### Running tests

Unit tests for this project can be run using `pytest .` from the main directory.
