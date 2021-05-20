### The following code is derived from:
### https://github.com/GiggleLiu/QuantumCircuitBornMachine/blob/master/LICENSE
### The original resource is under MIT license, which is pasted below for convenience

# MIT License
# Copyright (c) 2017 Leo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import TYPE_CHECKING, Dict, List, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from zquantum.core.bitstring_distribution import BitstringDistribution


def compute_rbf_kernel(x_i: np.ndarray, y_j: np.ndarray, sigma: float) -> np.ndarray:
    """Compute the gaussian (RBF) kernel matrix K, with K_ij = exp(-gamma |x_i - y_j|^2)
    and gamma = 1/(2*sigma).

    Args:
        x_i: Samples A (integers).
        y_j: Samples B (integers).
        sigma: The bandwidth of the gaussian kernel.

    Returns:
        np.ndarray: The gaussian kernel matrix.
    """
    exponent = np.abs(x_i[:, None] - y_j[None, :]) ** 2
    try:
        gamma = 1.0 / (2 * sigma)
    except ZeroDivisionError as error:
        print("Handling run-time error:", error)
        raise
    kernel_matrix = np.exp(-gamma * exponent)
    return kernel_matrix


def compute_multi_rbf_kernel(
    x_i: np.ndarray, y_j: np.ndarray, sigmas: Sequence[float]
) -> np.ndarray:
    """Compute the multi-gaussian (RBF) kernel matrix K, with

    K_ij = 1/N * Sum_n [exp(-gamma_n |x_i - y_j|^2)]

    with n = 1,...,N and gamma = 1/(2*sigma).

    Args:
        x_i: Samples A (integers).
        y_j: Samples B (integers).
        sigmas: The list of bandwidths of the multi-gaussian kernel.

    Returns:
        np.ndarray: The gaussian kernel matrix.
    """
    exponent = np.abs(x_i[:, None] - y_j[None, :]) ** 2
    kernel_matrix = np.zeros(exponent.shape)
    for sigma in sigmas:
        try:
            gamma = 1.0 / (2 * sigma)
        except ZeroDivisionError as error:
            print("Handling run-time error:", error)
            raise
        kernel_matrix += np.exp(-gamma * exponent)
    return kernel_matrix / len(sigmas)


def compute_mmd(
    target_distribution: "BitstringDistribution",
    measured_distribution: "BitstringDistribution",
    distance_measure_parameters: Dict,
) -> float:
    """Compute the squared Maximum Mean Discrepancy (MMD) distance measure between
    between a target bitstring distribution and a measured bitstring distribution.
    Reference: arXiv.1804.04168.

        Args:
            target_distribution: The target bitstring probability distribution.
            measured_distribution: The measured bitstring probability distribution.

            distance_measure_parameters:
                sigma (float/np.array): the bandwidth parameter used to compute the
                    single/multi gaussian kernel. The default value is 1.0.

        Returns:
            The value of the maximum mean discrepancy.
    """

    sigma = distance_measure_parameters.get("sigma", 1.0)
    target_keys = target_distribution.distribution_dict.keys()
    measured_keys = measured_distribution.distribution_dict.keys()
    all_keys = set(target_keys).union(measured_keys)

    target_values = []
    measured_values = []
    for bitstring in all_keys:
        # Add 0 to the values list whenever a bistrings isn't found among the keys.
        target_values.append(target_distribution.distribution_dict.get(bitstring, 0))
        measured_values.append(
            measured_distribution.distribution_dict.get(bitstring, 0)
        )

    basis = np.asarray([int(item, 2) for item in all_keys])  # bitstring to int
    if not hasattr(sigma, "__len__"):
        kernel_matrix = compute_rbf_kernel(basis, basis, sigma)
    else:
        kernel_matrix = compute_multi_rbf_kernel(basis, basis, sigma)

    diff = np.array(target_values) - np.array(measured_values)
    return diff.dot(kernel_matrix.dot(diff))
