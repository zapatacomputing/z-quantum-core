################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
include subtrees/z_quantum_actions/Makefile


mypy: clean
	@echo scanning files with mypy: Please be patient....
	$(PYTHON) -m mypy src

build-system-deps:
	$(PYTHON) -m pip install setuptools wheel "setuptools_scm>=6.0"
