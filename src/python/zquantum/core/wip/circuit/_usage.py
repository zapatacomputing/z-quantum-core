import numpy as np
import sympy
import pprint

from . import _builtin_gates as bg
from . import _gates as g


# circuit = g.Circuit([], 7)

# circuit += bg.X(0)
# circuit += bg.X(1)

# circuit += bg.RY(np.pi / 2)(0)
# circuit += bg.RY(sympy.Symbol("theta"))(1)

# circuit += g.OpaqueOperation(
#     transformation=lambda state_vector: state_vector * 2,
#     qubit_indices=(0, 3, 4)
# )


# pprint.pprint(circuit)



# my_ry = bg.RY(sympy.Symbol("theta"))

# my_ry2 = my_ry.bind({sympy.Symbol("theta"): 10})


circuit = g.Circuit([], 7)
theta = sympy.Symbol("theta")
circuit += bg.RY(theta)(0)
circuit += bg.RY(theta)(1)
circuit += bg.RY(theta)(2)
...
circuit += bg.RY(theta)(99)

circuit.bind({theta: np.pi})
