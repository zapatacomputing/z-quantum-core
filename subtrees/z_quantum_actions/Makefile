TOP_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
include $(TOP_DIR)/variables.mk

# This target will list the possible valid targets available.
default:
	@echo --------------------------------------------
	@echo '=> No target chosen: Choose from the following:'
	@echo
	@grep -E '^\w+(\-default)?:' $(TOP_DIR)/$(firstword $(MAKEFILE_LIST)) \
	       | sed -r 's/-default//g; /default/d ; s/(.*)/\t make \1/g ; s/:.*$$//g'

PYTHON := $(shell which python3)
REPO := $(shell git config --get remote.origin.url)
PYTHON_MOD := $(shell find src -maxdepth 3 -mindepth 3 -type d | sed '/.*cache/d; s/src\/python\/// ; s/\//./')

ifeq ($(PYTHON),)
$(error "PYTHON=$(PYTHON)")
else
$(info -------------------------------------------------------------------------------)
$(info You are using PYTHON: $(PYTHON))
$(info Python Version: $(shell $(PYTHON) --version))
$(info Repository: $(REPO))
$(info Python Modules Covered: $(PYTHON_MOD))
$(info -------------------------------------------------------------------------------)
endif

# Clean out all Pythonic cruft
clean-default:
	@find . -regex '^.*\(__pycache__\|\.py[co]\)$$' -delete;
	@find . -type d -name __pycache__ -exec rm -r {} \+
	@find . -type d -name '*.egg-info' -exec rm -rf {} +
	@find . -type d -name .mypy_cache -exec rm -r {} \+
	@rm -rf .pytest_cache;
	@rm -rf tests/.pytest_cache;
	@rm -rf dist build
	@rm -f .coverage*
	@echo Finished cleaning out pythonic cruft...

install-default: clean
	$(PYTHON) -m pip install --upgrade pip==21.2.4
	$(PYTHON) -m pip install .

# Renamed to develop to distinguish from dev branch
develop-default: clean
	$(PYTHON) -m pip install -e .[develop]

github_actions-default: clean
	$(PYTHON) -m pip install --user -e .[develop]

flake8-default: clean
	@flake8 --ignore=E203,E266,F401,W503 --max-line-length=88

mypy-default: clean
	@echo scanning files with mypy: Please be patient....
	@mypy --ignore-missing-imports --namespace-packages src

black-default: clean
	@black --check src tests

isort-default: clean
	@isort --check --profile black src tests

test-default:
	@pytest -m "not integration" tests

coverage-default:
	pytest -m "not integration" \
		--cov=src \
		--cov-fail-under=$(MIN_COVERAGE) tests \
		--no-cov-on-fail \
		--cov-report term-missing \
		&& echo Code coverage Passed the $(MIN_COVERAGE)% mark!


style-default: flake8 mypy black isort
	@echo This project passes style!

muster-default: style coverage
	@echo This project passes muster!

# This is what converts the -default targets into base target names.
# Do not remove!!!
%: %-default
	@true
