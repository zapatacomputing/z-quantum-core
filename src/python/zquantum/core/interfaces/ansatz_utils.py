from functools import wraps


class _InvalidatingSetter:
    """Setter descriptor that sets target object's _parametrized_circuit to None.

    The descriptor uses __get__ and __set__ methods. Both of them accept ansatz as a
    first argument (in this case).
    We just forward the __get__, but in __set__ we set obj._parametrized_circuit to None.
    """

    def __init__(self, target):
        self.target = target

    def __get__(self, ansatz, obj_type):
        return self.target.__get__(ansatz, obj_type)

    def __set__(self, ansatz, new_obj):
        self.target.__set__(ansatz, new_obj)
        ansatz._parametrized_circuit = None


def invalidates_parametrized_circuit(target):
    """
    Make given target (either property or method) invalidate ansatz's circuit.
    It can be used as a decorator, when for some reason `ansatz_property` shouldn't be used.
    """
    if isinstance(target, property):
        # If we are dealing with a property, return our modified descriptor.
        return _InvalidatingSetter(target)
    else:
        # Methods are functions that take instance as a first argument
        # They only change to "bound" methods once the object is instantiated
        # Therefore, we are decorating a function of signature _function(ansatz, ...)
        @wraps(target)
        def _wrapper(ansatz, *args, **kwargs):
            # Pass through the arguments, store the returned value for later use
            return_value = target(ansatz, *args, **kwargs)
            # Invalidate circuit
            ansatz._parametrized_circuit = None
            # Return original result
            return return_value

        return _wrapper


class DynamicProperty:
    """A shortcut to create a getter-setter descriptor with one liners."""

    def __init__(self, name: str, default_value=None):
        self.default_value = default_value
        self.name = name

    @property
    def attrname(self):
        return f"_{self.name}"

    def __get__(self, obj, obj_type):
        if not hasattr(obj, self.attrname):
            setattr(obj, self.attrname, self.default_value)
        return getattr(obj, self.attrname)

    def __set__(self, obj, new_obj):
        setattr(obj, self.attrname, new_obj)


def ansatz_property(name: str, default_value=None):
    return _InvalidatingSetter(DynamicProperty(name, default_value))
