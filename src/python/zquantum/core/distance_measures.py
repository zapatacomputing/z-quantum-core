### The following code is derived from: https://github.com/GiggleLiu/QuantumCircuitBornMachine/blob/master/LICENSE
### The original resource is under MIT license, which is pasted below for convenience >>>
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

import numpy as np

# TODO: Mixture of gaussian kernels?


def compute_rbf_kernel(x_i, y_j, float):
    """ Compute the gaussian (RBF) kernel matrix K, with K_ij = exp(-gamma |x_i - y_j|^2) and gamma = 1/(2*sigma).

        Args:
            x_i (np.array): Samples A (integers).
            y_j (np.array): Samples B (integers).
            sigma (float): The bandwidth of the gaussian kernel.

        Returns:
            np.ndarray: The gaussian kernel matrix.
    """
    exponent = np.abs(x_i[:, None] - y_j[None, :]) ** 2
    gamma = 1.0 / (2 * sigma)
    K = np.exp(-gamma * exponent)
    return K


def compute_mmd(
    target_distribution, measured_distribution, distance_measure_parameters,
):
    """ Compute the squared Maximum Mean Discrepancy (MMD) distance measure between between a target bitstring distribution
    and a measured bitstring distribution.
    Reference: arXiv.1804.04168.

        Args:
            target_distribution (BitstringDistribution): The target bitstring probability distribution.
            measured_distribution (BitstringDistribution): The measured bitstring probability distribution.
            distance_measure_parameters (dict): dictionary containing the relevant parameters for the clipped negative log likelihood, i.e.:
                                                - sigma: the bandwidth parameter used to compute the gaussian kernel.
        Returns:
            float: The value of the maximum mean discrepancy.
    """

    sigma = distance_measure_parameters.get("sigma", 1)
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
    K = compute_rbf_kernel(basis, basis, sigma)
    diff = np.array(target_values) - np.array(measured_values)
    return diff.dot(K.dot(diff))
