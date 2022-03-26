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
OpenFermion

For more information, examples, or tutorials visit our website:

www.openfermion.org
"""
# isort: skip_file


from zquantum.core.openfermion.chem import (
    MolecularData,
    angstroms_to_bohr,
    bohr_to_angstroms,
    geometry_from_file,
    load_molecular_hamiltonian,
    make_reduced_hamiltonian,
    name_molecule,
)

from zquantum.core.openfermion.hamiltonians import (
    fermi_hubbard,
    interaction_operator,
)

from zquantum.core.openfermion.linalg import (
    sparse_eigenspectrum,
    kronecker_operators,
    jw_number_restrict_operator,
    jw_number_indices,
    jw_hartree_fock_state,
    jw_configuration_state,
    jordan_wigner_sparse,
    wrapped_kronecker,
    jordan_wigner_ladder_sparse,
    eigenspectrum,
    expectation,
    get_density_matrix,
    get_ground_state,
    get_number_preserving_sparse_operator,
    get_sparse_operator,
    inner_product,
    jw_get_ground_state_at_particle_number,
    qubit_operator_sparse,
)

from zquantum.core.openfermion.measurements import get_interaction_rdm
from zquantum.core.openfermion.ops import (
    FermionOperator,
    InteractionOperator,
    InteractionOperatorError,
    InteractionRDM,
    InteractionRDMError,
    IsingOperator,
    PolynomialTensor,
    PolynomialTensorError,
    QubitOperator,
    SymbolicOperator,
    general_basis_change,
)

# Shifted here to fix circular dependencies
from zquantum.core.openfermion.circuits import (
    uccsd_convert_amplitude_format,
    uccsd_generator,
    uccsd_singlet_generator,
    uccsd_singlet_get_packed_amplitudes,
    uccsd_singlet_paramsize,
)

from zquantum.core.openfermion.transforms import (
    jordan_wigner_one_body,
    jordan_wigner_two_body,
    bravyi_kitaev,
    check_no_sympy,
    chemist_ordered,
    freeze_orbitals,
    get_fermion_operator,
    get_interaction_operator,
    get_molecular_data,
    jordan_wigner,
    normal_ordered,
    normal_ordered_ladder_term,
    prune_unused_indices,
    reorder,
    reverse_jordan_wigner,
)
from zquantum.core.openfermion.utils import (
    OperatorSpecificationError,
    OperatorUtilsError,
    anticommutator,
    commutator,
    count_qubits,
    double_commutator,
    down_index,
    get_file_path,
    hermitian_conjugated,
    is_hermitian,
    is_identity,
    load_operator,
    save_operator,
    up_index,
    up_then_down,
)

from zquantum.core.openfermion.zapata_utils import (
    # _utils.py
    get_qubitop_from_matrix,
    get_qubitop_from_coeffs_and_labels,
    generate_random_qubitop,
    evaluate_qubit_operator,
    evaluate_qubit_operator_list,
    reverse_qubit_order,
    get_expectation_value,
    change_operator_type,
    get_fermion_number_operator,
    get_diagonal_component,
    get_polynomial_tensor,
    create_circuits_from_qubit_operator,
    get_ground_state_rdm_from_qubit_op,
    remove_inactive_orbitals,
    hf_rdm,
    # _io.py
    convert_interaction_op_to_dict,
    convert_dict_to_interaction_op,
    load_interaction_operator,
    save_interaction_operator,
    convert_dict_to_qubitop,
    convert_qubitop_to_dict,
    convert_dict_to_operator,
    save_qubit_operator,
    load_qubit_operator,
    save_qubit_operator_set,
    load_qubit_operator_set,
    get_pauli_strings,
    convert_isingop_to_dict,
    convert_dict_to_isingop,
    load_ising_operator,
    save_ising_operator,
    save_parameter_grid_evaluation,
    convert_interaction_rdm_to_dict,
    convert_dict_to_interaction_rdm,
    load_interaction_rdm,
    save_interaction_rdm,
)
