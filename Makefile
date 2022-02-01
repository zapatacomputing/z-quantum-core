include subtrees/z_quantum_actions/Makefile


mypy: clean
	@echo scanning files with mypy: Please be patient....
	$(PYTHON) -m mypy src
