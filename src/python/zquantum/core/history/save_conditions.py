"""Save conditions possible to use with recorder."""


def always(value, params, call_number):
    """Default save condition: save always."""
    return True


def every_nth(n):
    """Save condition: every n-th step, counting from zero-th one.

    Note: this is factory function, i.e. it returns the actual save condition for given n.
    """
    def _save_condition(value, params, call_number):
        return call_number % n == 0

    return _save_condition
