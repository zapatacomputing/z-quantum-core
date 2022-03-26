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
"""Commonly used operators (mainly instances of SymbolicOperator)."""
from typing import Optional

from zquantum.core.openfermion.ops.operators import FermionOperator


def number_operator(
    n_modes: int, mode: Optional[int] = None, coefficient=1.0, parity: int = -1
) -> FermionOperator:
    """Return a fermionic number operator.

    Args:
        n_modes (int): The number of modes in the system.
        mode (int, optional): The mode on which to return the number
            operator. If None, return total number operator on all sites.
        coefficient (float): The coefficient of the term.
        parity (int): Returns the fermionic number operator
                    if parity=-1 (default),
                    and returns the bosonic number operator
                    if parity=1.
    Returns:
        operator (FermionOperator)
    """

    if parity == -1:
        Op = FermionOperator
    elif parity == 1:
        # Op = BosonOperator
        raise ValueError("BosonOperator not available")
    else:
        raise ValueError("Invalid parity value: {}".format(parity))

    if mode is None:
        operator = Op()
        for m in range(n_modes):
            operator += number_operator(n_modes, m, coefficient, parity)
    else:
        operator = Op(((mode, 1), (mode, 0)), coefficient)
    return operator
