import numpy as np
import sympy
import pprint

from . import _builtin_gates as g


circuit = [
    g.X(0),
    g.X(1),
    g.RY(np.pi / 2)(0),
    g.RY(sympy.Symbol("theta"))(1),
]

pprint.pprint(circuit)
