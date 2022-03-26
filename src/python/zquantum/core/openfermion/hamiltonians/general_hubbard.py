#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""This module constructs Hamiltonians for the multiband Fermi-Hubbard model.
"""

from collections import namedtuple

from zquantum.core.openfermion.ops.operators import FermionOperator

TunnelingParameter = namedtuple(
    "TunnelingParameter", ("edge_type", "dofs", "coefficient")
)
InteractionParameter = namedtuple(
    "InteractionParameter", ("edge_type", "dofs", "coefficient", "spin_pairs")
)
PotentialParameter = namedtuple("PotentialParameter", ("dof", "coefficient"))


def number_operator(i, coefficient=1.0, particle_hole_symmetry=False):
    op = FermionOperator(((i, 1), (i, 0)), coefficient)
    if particle_hole_symmetry:
        op -= FermionOperator((), 0.5)
    return op


def interaction_operator(i, j, coefficient=1.0, particle_hole_symmetry=False):
    return number_operator(
        i, coefficient, particle_hole_symmetry=particle_hole_symmetry
    ) * number_operator(j, particle_hole_symmetry=particle_hole_symmetry)
