# z-quantum-core

[![codecov](https://codecov.io/gh/zapatacomputing/z-quantum-core/branch/master/graph/badge.svg?token=ZWVLF7O2OQ)](https://codecov.io/gh/zapatacomputing/z-quantum-core)


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
    load_array,
    save_circuit,
)

ansatz = load_circuit_template("ansatz.json")
params = load_array("params.json")
circuit = build_ansatz_circuit(ansatz, params)
save_circuit(circuit, "circuit.json")
```

Even though it's intended to be used with Orquestra, `z-quantum-core` can be also used as a standalone Python module.
To install it, you just need to run `pip install -e .` from the main directory. )

## Development and Contribution

To install the development version, run `pip install -e '.[dev]'` from the main directory. (if using MacOS, you will need single quotes around the []. If using windows, or Linux, you might not need the quotes).

We use [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) docstring format. If you'd like to specify types please use [PEP 484](https://www.python.org/dev/peps/pep-0484/) type hints instead adding them to docstrings.

There are codestyle-related [Github Actions](.github/workflows/style.yml) running for PRs to `master` or `dev`. 

Additionally, you can set up our [pre-commit hooks](.pre-commit-config.yaml) with `pre-commit install` . It checks for minor errors right before commiting or pushing code for quick feedback. More info [here](https://pre-commit.com). Note that if needed, you can skip these checks with the `--no-verify` option, i.e. `git commit -m "Add quickfix, prod is on fire" --no-verify`.

- If you'd like to report a bug/issue please create a new issue in this repository.
- If you'd like to contribute, please create a pull request.

### Running tests

Unit tests for this project can be run using `pytest .` from the main directory.
