# https://stackoverflow.com/questions/24722212/python-cant-find-module-in-the-same-folder
import sys
import os

sys.path.append("/app/step/qe-cli-wrapper/src/python")
sys.path.append("/app/step/z-quantum-core/src/python")
sys.path.append("/app/step/z-quantum-vqe/src/python")
sys.path.append("/app/step/z-quantum-optimizers/src/python")
sys.path.append("/app/step/diatomic-molecule/src/python")
sys.path.append("/app/step/qe-psi4/src/python")
sys.path.append("/app/step/qe-forest/src/python")
sys.path.append("/app/step/qe-qulacs/src/python")
sys.path.append(os.path.dirname(__file__))

import numpy as np

from zquantum.core.cost_function import AnsatzBasedCostFunction
from zquantum.vqe.singlet_uccsd import SingletUCCSDAnsatz
from zquantum.core.estimator import BasicEstimator, ExactEstimator
from qequlacs import QulacsSimulator
from zquantum.optimizers import ScipyOptimizer
import qe.sdk.v1 as qe
from zquantum.core.openfermion import convert_dict_to_interaction_op

from openfermion import (
    jordan_wigner,
    bravyi_kitaev,
    get_fermion_operator,
    QubitOperator,
    SymbolicOperator,
)


@qe.step(
    resource_def=qe.ResourceDefiniton(
        cpu="1000m",
        mem="1000MB",
        disk="1G",
    ),
)
def get_molecule() -> dict:
    return {
        "sites": [
            {"species": "H", "x": 0, "y": 0, "z": 0},
            {"species": "H", "x": 0, "y": 0, "z": 0.7},
        ]
    }


# def run_psi4(
#     basis,
#     method,
#     reference,
#     geometry,
#     freeze_core=False,
#     charge=0,
#     multiplicity=1,
#     save_hamiltonian=False,
#     save_rdms=False,
#     n_active_extract=None,
#     n_occupied_extract=None,
#     freeze_core_extract=False,
#     nthreads=1,
#     options=None,
#     wavefunction=None,
# ):
#     from qepsi4 import run_psi4 as _run_psi4

#     # try:
#     #     os.mkdir("/tmp/scr")
#     # except:
#     #     pass
#     os.environ["PSI_SCRATCH"] = "/tmp/scr"

#     return _run_psi4(
#         geometry,
#         basis=basis,
#         multiplicity=multiplicity,
#         charge=charge,
#         method=method,
#         reference=reference,
#         freeze_core=freeze_core,
#         save_hamiltonian=save_hamiltonian,
#         save_rdms=save_rdms,
#         options=options,
#         n_active_extract=n_active_extract,
#         n_occupied_extract=n_occupied_extract,
#         freeze_core_extract=freeze_core_extract,
#     )


@qe.step(
    resource_def=qe.ResourceDefiniton(
        cpu="1000m",
        mem="1000MB",
        disk="1G",
    ),
)
def get_hamiltonian(molecule) -> dict:
    return {
        "two_body_tensor": {
            "real": [
                [
                    [
                        [0.4715467733370239, 0.0, 0.08649174882784871, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.08649174882784866, 0.0, 0.07265829776863195, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.4715467733370239, 0.0, 0.08649174882784871, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.08649174882784866, 0.0, 0.07265829776863195, 0.0],
                    ],
                    [
                        [0.08649174882784867, 0.0, 0.07265829776863197, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.3299862360865468, 0.0, -0.018695382803028626, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.08649174882784867, 0.0, 0.07265829776863197, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.3299862360865468, 0.0, -0.018695382803028626, 0.0],
                    ],
                ],
                [
                    [
                        [0.0, 0.4715467733370239, 0.0, 0.08649174882784871],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.08649174882784866, 0.0, 0.07265829776863195],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.4715467733370239, 0.0, 0.08649174882784871],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.08649174882784866, 0.0, 0.07265829776863195],
                    ],
                    [
                        [0.0, 0.08649174882784867, 0.0, 0.07265829776863197],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.3299862360865468, 0.0, -0.018695382803028626],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.08649174882784867, 0.0, 0.07265829776863197],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.3299862360865468, 0.0, -0.018695382803028626],
                    ],
                ],
                [
                    [
                        [0.08649174882784857, 0.0, 0.3299862360865467, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.07265829776863186, 0.0, -0.018695382803028716, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.08649174882784857, 0.0, 0.3299862360865467, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.07265829776863186, 0.0, -0.018695382803028716, 0.0],
                    ],
                    [
                        [0.07265829776863186, 0.0, -0.018695382803028807, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [-0.018695382803028772, 0.0, 0.37624725093034594, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.07265829776863186, 0.0, -0.018695382803028807, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [-0.018695382803028772, 0.0, 0.37624725093034594, 0.0],
                    ],
                ],
                [
                    [
                        [0.0, 0.08649174882784857, 0.0, 0.3299862360865467],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.07265829776863186, 0.0, -0.018695382803028716],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.08649174882784857, 0.0, 0.3299862360865467],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.07265829776863186, 0.0, -0.018695382803028716],
                    ],
                    [
                        [0.0, 0.07265829776863186, 0.0, -0.018695382803028807],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, -0.018695382803028772, 0.0, 0.37624725093034594],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.07265829776863186, 0.0, -0.018695382803028807],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, -0.018695382803028772, 0.0, 0.37624725093034594],
                    ],
                ],
            ]
        },
        "constant": {"real": 1.3656186028129031},
        "schema": "io-ZapOS-v1alpha1-interaction_op",
        "one_body_tensor": {
            "real": [
                [-2.5753433475624083, 0.0, -0.17296506308021864, 0.0],
                [0.0, -2.5753433475624083, 0.0, -0.17296506308021864],
                [-0.17296506308021853, 0.0, -1.3476333096577398, 0.0],
                [0.0, -0.17296506308021853, 0.0, -1.3476333096577398],
            ]
        },
    }
    # return run_psi4(
    #     basis="STO-3G",
    #     method="scf",
    #     reference="rhf",
    #     geometry=molecule,
    #     freeze_core=False,
    #     charge=0,
    #     multiplicity=1,
    #     save_hamiltonian=True,
    #     nthreads=4,
    # )


@qe.step(
    resource_def=qe.ResourceDefiniton(
        cpu="1000m",
        mem="1000MB",
        disk="1G",
    ),
)
def get_qubit_hamiltonian(hamiltonian) -> dict:
    return jordan_wigner(convert_dict_to_interaction_op(hamiltonian))


@qe.step(
    resource_def=qe.ResourceDefiniton(
        cpu="1000m",
        mem="1000MB",
        disk="1G",
    ),
)
def optimize_circuit(qubit_hamiltonian) -> dict:
    ansatz = SingletUCCSDAnsatz(
        number_of_layers=1,
        number_of_spatial_orbitals=2,
        number_of_alpha_electrons=1,
        transformation="Jordan-Wigner",
    )
    backend = QulacsSimulator()

    estimator = ExactEstimator()
    optimizer = ScipyOptimizer(method="L-BFGS-B")
    cost_function = AnsatzBasedCostFunction(
        qubit_hamiltonian, ansatz, backend, estimator
    )
    initial_params = np.array([0.01, -0.01])

    # When
    opt_results = optimizer.minimize(cost_function, initial_params)
    return opt_results


@qe.workflow(
    name="hello-optimizers",
    import_defs=[
        qe.GitImportDefinition.get_current_repo_and_branch(),
        qe.Z.Quantum.Vqe(),
        qe.Z.Quantum.Optimizers(),
        qe.GitImportDefinition(
            "git@github.com:zapatacomputing/tutorial-1-diatomic-molecule.git"
        ),
        qe.GitImportDefinition("git@github.com:zapatacomputing/qe-psi4.git"),
        qe.GitImportDefinition("git@github.com:zapatacomputing/qe-forest.git"),
        qe.GitImportDefinition("git@github.com:zapatacomputing/qe-qulacs.git"),
    ],
)
def opt_workflow() -> qe.StepDefinition:
    return optimize_circuit(get_qubit_hamiltonian(get_hamiltonian(get_molecule())))


if __name__ == "__main__":
    wf: qe.WorkflowDefinition = opt_workflow()
    result = wf.local_run()  # pylint: disable=no-member
    wf.print_workflow()  # pylint: disable=no-member
    wf.submit()  # pylint: disable=no-member

    # result = qe.load_workflowresult("hello-optimizers")
    # print(result)