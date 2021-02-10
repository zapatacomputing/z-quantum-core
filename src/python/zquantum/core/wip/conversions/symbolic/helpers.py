"""Helper functions for use with symbolic expressions."""
from functools import reduce


def reduction(operator):
    def _reduction(*args):
        return reduce(operator, args)
    return _reduction
