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
"""
test coverage for performance benchmarks. Check type returns.
"""
from .performance_benchmarks import (
    benchmark_commutator_diagonal_coulomb_operators_2D_spinless_jellium,
    benchmark_fermion_math_and_normal_order,
    benchmark_jordan_wigner_sparse,
    benchmark_linear_qubit_operator,
    benchmark_molecular_operator_jordan_wigner,
    run_diagonal_commutator,
    run_fermion_math_and_normal_order,
    run_jordan_wigner_sparse,
    run_linear_qubit_operator,
    run_molecular_operator_jordan_wigner,
)


def test_run_jw_speed():
    timing = benchmark_molecular_operator_jordan_wigner(2)
    assert isinstance(timing, float)


def test_fermion_normal_order():
    r1, r2 = benchmark_fermion_math_and_normal_order(4, 2, 2)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_jw_sparse():
    r1 = benchmark_jordan_wigner_sparse(4)
    assert isinstance(r1, float)


def test_linear_qop():
    r1, r2 = benchmark_linear_qubit_operator(3, 2)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_comm_diag_coulomb():
    r1, r2 = benchmark_commutator_diagonal_coulomb_operators_2D_spinless_jellium(4)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_mol_jw():
    r1 = run_molecular_operator_jordan_wigner(3)
    assert isinstance(r1, float)


def test_run_fermion_no():
    r1, r2 = run_fermion_math_and_normal_order(2)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_jw_sparse_time():
    r1 = run_jordan_wigner_sparse(2)
    assert isinstance(r1, float)


def test_run_linear_qop():
    r1, r2 = run_linear_qubit_operator(2)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_run_diag_comm():
    r1, r2 = run_diagonal_commutator(4)
    assert isinstance(r1, float)
    assert isinstance(r2, float)
