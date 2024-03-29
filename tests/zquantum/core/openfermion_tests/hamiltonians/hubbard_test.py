#   Copyright 2017 The OpenFermion Developers
#   Modifications copyright 2022 Zapata Computing, Inc. for compatibility reasons.
#
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
"""Tests for Hubbard model module."""

from zquantum.core.openfermion.hamiltonians import fermi_hubbard


def test_fermi_hubbard_1x3_spinless():
    hubbard_model = fermi_hubbard(1, 3, 1.0, 4.0, chemical_potential=0.5, spinless=True)
    assert (
        str(hubbard_model).strip()
        == """
-0.5 [0^ 0] +
4.0 [0^ 0 1^ 1] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [1^ 0] +
-0.5 [1^ 1] +
4.0 [1^ 1 2^ 2] +
-1.0 [1^ 2] +
-1.0 [2^ 0] +
-1.0 [2^ 1] +
-0.5 [2^ 2] +
4.0 [2^ 2 0^ 0]
""".strip()
    )


def test_fermi_hubbard_3x1_spinless():
    hubbard_model = fermi_hubbard(3, 1, 1.0, 4.0, chemical_potential=0.5, spinless=True)
    assert (
        str(hubbard_model).strip()
        == """
-0.5 [0^ 0] +
4.0 [0^ 0 1^ 1] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [1^ 0] +
-0.5 [1^ 1] +
4.0 [1^ 1 2^ 2] +
-1.0 [1^ 2] +
-1.0 [2^ 0] +
-1.0 [2^ 1] +
-0.5 [2^ 2] +
4.0 [2^ 2 0^ 0]
""".strip()
    )


def test_fermi_hubbard_2x2_spinless():
    hubbard_model = fermi_hubbard(2, 2, 1.0, 4.0, chemical_potential=0.5, spinless=True)
    assert (
        str(hubbard_model).strip()
        == """
-0.5 [0^ 0] +
4.0 [0^ 0 1^ 1] +
4.0 [0^ 0 2^ 2] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [1^ 0] +
-0.5 [1^ 1] +
4.0 [1^ 1 3^ 3] +
-1.0 [1^ 3] +
-1.0 [2^ 0] +
-0.5 [2^ 2] +
4.0 [2^ 2 3^ 3] +
-1.0 [2^ 3] +
-1.0 [3^ 1] +
-1.0 [3^ 2] +
-0.5 [3^ 3]
""".strip()
    )


def test_fermi_hubbard_2x3_spinless():
    hubbard_model = fermi_hubbard(2, 3, 1.0, 4.0, chemical_potential=0.5, spinless=True)
    assert (
        str(hubbard_model).strip()
        == """
-0.5 [0^ 0] +
4.0 [0^ 0 1^ 1] +
4.0 [0^ 0 2^ 2] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 4] +
-1.0 [1^ 0] +
-0.5 [1^ 1] +
4.0 [1^ 1 3^ 3] +
-1.0 [1^ 3] +
-1.0 [1^ 5] +
-1.0 [2^ 0] +
-0.5 [2^ 2] +
4.0 [2^ 2 3^ 3] +
4.0 [2^ 2 4^ 4] +
-1.0 [2^ 3] +
-1.0 [2^ 4] +
-1.0 [3^ 1] +
-1.0 [3^ 2] +
-0.5 [3^ 3] +
4.0 [3^ 3 5^ 5] +
-1.0 [3^ 5] +
-1.0 [4^ 0] +
-1.0 [4^ 2] +
-0.5 [4^ 4] +
4.0 [4^ 4 0^ 0] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 5] +
-1.0 [5^ 1] +
-1.0 [5^ 3] +
-1.0 [5^ 4] +
-0.5 [5^ 5] +
4.0 [5^ 5 1^ 1]
""".strip()
    )


def test_fermi_hubbard_3x2_spinless():
    hubbard_model = fermi_hubbard(3, 2, 1.0, 4.0, chemical_potential=0.5, spinless=True)
    assert (
        str(hubbard_model).strip()
        == """
-0.5 [0^ 0] +
4.0 [0^ 0 1^ 1] +
4.0 [0^ 0 3^ 3] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 3] +
-1.0 [1^ 0] +
-0.5 [1^ 1] +
4.0 [1^ 1 2^ 2] +
4.0 [1^ 1 4^ 4] +
-1.0 [1^ 2] +
-1.0 [1^ 4] +
-1.0 [2^ 0] +
-1.0 [2^ 1] +
-0.5 [2^ 2] +
4.0 [2^ 2 0^ 0] +
4.0 [2^ 2 5^ 5] +
-1.0 [2^ 5] +
-1.0 [3^ 0] +
-0.5 [3^ 3] +
4.0 [3^ 3 4^ 4] +
-1.0 [3^ 4] +
-1.0 [3^ 5] +
-1.0 [4^ 1] +
-1.0 [4^ 3] +
-0.5 [4^ 4] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 5] +
-1.0 [5^ 2] +
-1.0 [5^ 3] +
-1.0 [5^ 4] +
-0.5 [5^ 5] +
4.0 [5^ 5 3^ 3]
""".strip()
    )


def test_fermi_hubbard_3x3_spinless():
    hubbard_model = fermi_hubbard(3, 3, 1.0, 4.0, chemical_potential=0.5, spinless=True)
    assert (
        str(hubbard_model).strip()
        == """
-0.5 [0^ 0] +
4.0 [0^ 0 1^ 1] +
4.0 [0^ 0 3^ 3] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 3] +
-1.0 [0^ 6] +
-1.0 [1^ 0] +
-0.5 [1^ 1] +
4.0 [1^ 1 2^ 2] +
4.0 [1^ 1 4^ 4] +
-1.0 [1^ 2] +
-1.0 [1^ 4] +
-1.0 [1^ 7] +
-1.0 [2^ 0] +
-1.0 [2^ 1] +
-0.5 [2^ 2] +
4.0 [2^ 2 0^ 0] +
4.0 [2^ 2 5^ 5] +
-1.0 [2^ 5] +
-1.0 [2^ 8] +
-1.0 [3^ 0] +
-0.5 [3^ 3] +
4.0 [3^ 3 4^ 4] +
4.0 [3^ 3 6^ 6] +
-1.0 [3^ 4] +
-1.0 [3^ 5] +
-1.0 [3^ 6] +
-1.0 [4^ 1] +
-1.0 [4^ 3] +
-0.5 [4^ 4] +
4.0 [4^ 4 5^ 5] +
4.0 [4^ 4 7^ 7] +
-1.0 [4^ 5] +
-1.0 [4^ 7] +
-1.0 [5^ 2] +
-1.0 [5^ 3] +
-1.0 [5^ 4] +
-0.5 [5^ 5] +
4.0 [5^ 5 3^ 3] +
4.0 [5^ 5 8^ 8] +
-1.0 [5^ 8] +
-1.0 [6^ 0] +
-1.0 [6^ 3] +
-0.5 [6^ 6] +
4.0 [6^ 6 0^ 0] +
4.0 [6^ 6 7^ 7] +
-1.0 [6^ 7] +
-1.0 [6^ 8] +
-1.0 [7^ 1] +
-1.0 [7^ 4] +
-1.0 [7^ 6] +
-0.5 [7^ 7] +
4.0 [7^ 7 1^ 1] +
4.0 [7^ 7 8^ 8] +
-1.0 [7^ 8] +
-1.0 [8^ 2] +
-1.0 [8^ 5] +
-1.0 [8^ 6] +
-1.0 [8^ 7] +
-0.5 [8^ 8] +
4.0 [8^ 8 2^ 2] +
4.0 [8^ 8 6^ 6]
""".strip()
    )


def test_fermi_hubbard_2x2_spinful():
    hubbard_model = fermi_hubbard(
        2, 2, 1.0, 4.0, chemical_potential=0.5, magnetic_field=0.3, spinless=False
    )
    assert (
        str(hubbard_model).strip()
        == """
-0.8 [0^ 0] +
4.0 [0^ 0 1^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 4] +
-0.2 [1^ 1] +
-1.0 [1^ 3] +
-1.0 [1^ 5] +
-1.0 [2^ 0] +
-0.8 [2^ 2] +
4.0 [2^ 2 3^ 3] +
-1.0 [2^ 6] +
-1.0 [3^ 1] +
-0.2 [3^ 3] +
-1.0 [3^ 7] +
-1.0 [4^ 0] +
-0.8 [4^ 4] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 6] +
-1.0 [5^ 1] +
-0.2 [5^ 5] +
-1.0 [5^ 7] +
-1.0 [6^ 2] +
-1.0 [6^ 4] +
-0.8 [6^ 6] +
4.0 [6^ 6 7^ 7] +
-1.0 [7^ 3] +
-1.0 [7^ 5] +
-0.2 [7^ 7]
""".strip()
    )


def test_fermi_hubbard_2x3_spinful():
    hubbard_model = fermi_hubbard(
        2, 3, 1.0, 4.0, chemical_potential=0.5, magnetic_field=0.3, spinless=False
    )
    assert (
        str(hubbard_model).strip()
        == """
-0.8 [0^ 0] +
4.0 [0^ 0 1^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 4] +
-1.0 [0^ 8] +
-0.2 [1^ 1] +
-1.0 [1^ 3] +
-1.0 [1^ 5] +
-1.0 [1^ 9] +
-1.0 [2^ 0] +
-0.8 [2^ 2] +
4.0 [2^ 2 3^ 3] +
-1.0 [2^ 6] +
-1.0 [2^ 10] +
-1.0 [3^ 1] +
-0.2 [3^ 3] +
-1.0 [3^ 7] +
-1.0 [3^ 11] +
-1.0 [4^ 0] +
-0.8 [4^ 4] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 6] +
-1.0 [4^ 8] +
-1.0 [5^ 1] +
-0.2 [5^ 5] +
-1.0 [5^ 7] +
-1.0 [5^ 9] +
-1.0 [6^ 2] +
-1.0 [6^ 4] +
-0.8 [6^ 6] +
4.0 [6^ 6 7^ 7] +
-1.0 [6^ 10] +
-1.0 [7^ 3] +
-1.0 [7^ 5] +
-0.2 [7^ 7] +
-1.0 [7^ 11] +
-1.0 [8^ 0] +
-1.0 [8^ 4] +
-0.8 [8^ 8] +
4.0 [8^ 8 9^ 9] +
-1.0 [8^ 10] +
-1.0 [9^ 1] +
-1.0 [9^ 5] +
-0.2 [9^ 9] +
-1.0 [9^ 11] +
-1.0 [10^ 2] +
-1.0 [10^ 6] +
-1.0 [10^ 8] +
-0.8 [10^ 10] +
4.0 [10^ 10 11^ 11] +
-1.0 [11^ 3] +
-1.0 [11^ 7] +
-1.0 [11^ 9] +
-0.2 [11^ 11]
""".strip()
    )


def test_fermi_hubbard_2x2_spinful_phs():
    hubbard_model = fermi_hubbard(
        2,
        2,
        1.0,
        4.0,
        chemical_potential=0.5,
        magnetic_field=0.3,
        spinless=False,
        particle_hole_symmetry=True,
    )
    assert (
        str(hubbard_model).strip()
        == """
4.0 [] +
-2.8 [0^ 0] +
4.0 [0^ 0 1^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 4] +
-2.2 [1^ 1] +
-1.0 [1^ 3] +
-1.0 [1^ 5] +
-1.0 [2^ 0] +
-2.8 [2^ 2] +
4.0 [2^ 2 3^ 3] +
-1.0 [2^ 6] +
-1.0 [3^ 1] +
-2.2 [3^ 3] +
-1.0 [3^ 7] +
-1.0 [4^ 0] +
-2.8 [4^ 4] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 6] +
-1.0 [5^ 1] +
-2.2 [5^ 5] +
-1.0 [5^ 7] +
-1.0 [6^ 2] +
-1.0 [6^ 4] +
-2.8 [6^ 6] +
4.0 [6^ 6 7^ 7] +
-1.0 [7^ 3] +
-1.0 [7^ 5] +
-2.2 [7^ 7]
""".strip()
    )


def test_fermi_hubbard_2x2_spinful_aperiodic():
    hubbard_model = fermi_hubbard(
        2,
        2,
        1.0,
        4.0,
        chemical_potential=0.5,
        magnetic_field=0.3,
        spinless=False,
        periodic=False,
    )
    assert (
        str(hubbard_model).strip()
        == """
-0.8 [0^ 0] +
4.0 [0^ 0 1^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 4] +
-0.2 [1^ 1] +
-1.0 [1^ 3] +
-1.0 [1^ 5] +
-1.0 [2^ 0] +
-0.8 [2^ 2] +
4.0 [2^ 2 3^ 3] +
-1.0 [2^ 6] +
-1.0 [3^ 1] +
-0.2 [3^ 3] +
-1.0 [3^ 7] +
-1.0 [4^ 0] +
-0.8 [4^ 4] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 6] +
-1.0 [5^ 1] +
-0.2 [5^ 5] +
-1.0 [5^ 7] +
-1.0 [6^ 2] +
-1.0 [6^ 4] +
-0.8 [6^ 6] +
4.0 [6^ 6 7^ 7] +
-1.0 [7^ 3] +
-1.0 [7^ 5] +
-0.2 [7^ 7]
""".strip()
    )
