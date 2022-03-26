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
"""Tests for multiband Hubbard model module."""

import random
from enum import Enum

from zquantum.core.openfermion.hamiltonians.general_hubbard import number_operator
from zquantum.core.openfermion.ops import FermionOperator


class SpinPairs(Enum):
    """The spin orbitals corresponding to a pair of spatial orbitals."""

    SAME = 1
    ALL = 0
    DIFF = -1


def random_parameters(lattice, probability=0.5, distinguish_edges=False):
    parameters = {}
    edge_types = (
        ("onsite", "horizontal_neighbor", "vertical_neighbor")
        if distinguish_edges
        else ("onsite", "neighbor")
    )

    parameters["tunneling_parameters"] = [
        (edge_type, dofs, random.uniform(-1, 1))
        for edge_type in edge_types
        for dofs in lattice.dof_pairs_iter(edge_type == "onsite")
        if random.random() <= probability
    ]

    possible_spin_pairs = (
        (SpinPairs.ALL,) if lattice.spinless else (SpinPairs.SAME, SpinPairs.DIFF)
    )
    parameters["interaction_parameters"] = [
        (edge_type, dofs, random.uniform(-1, 1), spin_pairs)
        for edge_type in edge_types
        for spin_pairs in possible_spin_pairs
        for dofs in lattice.dof_pairs_iter(
            edge_type == "onsite" and spin_pairs in (SpinPairs.ALL, SpinPairs.SAME)
        )
        if random.random() <= probability
    ]

    parameters["potential_parameters"] = [
        (dof, random.uniform(-1, 1))
        for dof in lattice.dof_indices
        if random.random() <= probability
    ]

    if random.random() <= probability:
        parameters["magnetic_field"] = random.uniform(-1, 1)

    return parameters


def test_number_op():
    nop = FermionOperator(((0, 1), (0, 0)), coefficient=1.0)
    test_op = number_operator(0)
    assert test_op == nop

    test_op = number_operator(0, particle_hole_symmetry=True)
    assert test_op == nop - FermionOperator((), coefficient=0.5)
