include subtrees/z_quantum_actions/Makefile


coverage:
	PYTHONPATH="." $(PYTHON) -m pytest -m "not integration" \
		--cov=src \
		--cov-fail-under=$(MIN_COVERAGE) tests \
		--no-cov-on-fail \
		--cov-report xml \
		&& echo Code coverage Passed the $(MIN_COVERAGE)% mark!

mypy-default: clean
	@echo scanning files with mypy: Please be patient....
	@mypy src