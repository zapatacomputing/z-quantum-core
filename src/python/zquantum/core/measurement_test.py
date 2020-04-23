import unittest
import os
import numpy as np
from openfermion.ops import QubitOperator, IsingOperator
from .measurement import (ExpectationValues, Parities,
                    save_expectation_values, load_expectation_values,
                    save_wavefunction, load_wavefunction, sample_from_wavefunction,
                    save_parities, load_parities, get_parities_from_measurements, 
                    get_expectation_values_from_measurements, 
                    get_expectation_values_from_parities,
                    expectation_values_to_real, convert_bitstring_to_int)
from pyquil.wavefunction import Wavefunction

from .testing import create_random_wavefunction
from .utils import convert_bitstrings_to_tuples
from collections import Counter

class TestMeasurement(unittest.TestCase):

    def test_expectation_values_io(self):
        expectation_values = np.array([0., 0., -1.0])
        correlations = []
        correlations.append(np.array([[1., -1.],[-1., 1.]]))
        correlations.append(np.array([[1.]]))

        covariances = []
        covariances.append(np.array([[0.1, -0.1],[-0.1, 0.1]]))
        covariances.append(np.array([[0.1]]))

        expectation_values_object = ExpectationValues(expectation_values, correlations, covariances)

        save_expectation_values(expectation_values_object, 'expectation_values.json')
        expectation_values_object_loaded = load_expectation_values('expectation_values.json')

        self.assertTrue(np.allclose(expectation_values_object.values,
                        expectation_values_object_loaded.values))
        self.assertEqual(len(expectation_values_object.correlations),
                         len(expectation_values_object_loaded.correlations))
        self.assertEqual(len(expectation_values_object.covariances),
                         len(expectation_values_object_loaded.covariances))
        for i in range(len(expectation_values_object.correlations)):
            self.assertTrue(np.allclose(expectation_values_object.correlations[i],
                                        expectation_values_object_loaded.correlations[i]))
        for i in range(len(expectation_values_object.covariances)):
            self.assertTrue(np.allclose(expectation_values_object.covariances[i],
                                        expectation_values_object_loaded.covariances[i]))
        
        os.remove('expectation_values.json')

    def test_real_wavefunction_io(self):
        wf = Wavefunction([0, 1, 0, 0, 0, 0, 0, 0])
        save_wavefunction(wf, 'wavefunction.json')
        loaded_wf = load_wavefunction('wavefunction.json')
        self.assertTrue(np.allclose(wf.amplitudes, loaded_wf.amplitudes))
        os.remove('wavefunction.json')

    def test_imag_wavefunction_io(self):
        wf = Wavefunction([0, 1j, 0, 0, 0, 0, 0, 0])
        save_wavefunction(wf, 'wavefunction.json')
        loaded_wf = load_wavefunction('wavefunction.json')
        self.assertTrue(np.allclose(wf.amplitudes, loaded_wf.amplitudes))
        os.remove('wavefunction.json')

    def test_sample_from_wavefunction(self):
        wavefunction = create_random_wavefunction(4)

        samples = sample_from_wavefunction(wavefunction, 100000)
        sampled_dict = Counter(samples)

        sampled_probabilities = []
        for num in range(len(wavefunction)**2):
            bitstring = format(num, 'b')
            while(len(bitstring) < len(wavefunction)):
                bitstring = '0' + bitstring
            bitstring = bitstring[::-1]

            measurement = convert_bitstrings_to_tuples([bitstring])[0]

            sampled_probabilities.append(sampled_dict[measurement]/100000)

        probabilities = wavefunction.probabilities()
        for sampled_prob, exact_prob in zip(sampled_probabilities, probabilities):
            self.assertAlmostEqual(sampled_prob, exact_prob, 2)

    def test_parities_io(self):
        measurements = [(1, 0), (1, 0), (0, 1), (0, 0)]
        op = IsingOperator('[Z0] + [Z1] + [Z0 Z1]')
        parities = get_parities_from_measurements(measurements, op)
        save_parities(parities, 'parities.json')
        loaded_parities = load_parities('parities.json')
        self.assertTrue(np.allclose(parities.values, loaded_parities.values))
        self.assertEqual(len(parities.correlations), len(loaded_parities.correlations))
        for i in range(len(parities.correlations)):
            self.assertTrue(np.allclose(parities.correlations[i], loaded_parities.correlations[i]))
        os.remove('parities.json')

    def test_get_expectation_values_from_measurements(self):
        # Given
        measurements = [(0,1,0), (0,1,0), (0,0,0), (0,0,0), (1,1,1)]
        ising_operator = IsingOperator('10[] + [Z0 Z1] - 10[Z1 Z2]')
        target_expectation_values = np.array([10, 0.2, -2])
        # When
        expectation_values = get_expectation_values_from_measurements(measurements, ising_operator)
        # Then
        np.testing.assert_array_equal(expectation_values.values, target_expectation_values)


    def test_get_expectation_values_from_parities(self):
        parities = Parities(values=np.array([[18, 50], [120, 113], [75, 26]]))
        expectation_values = get_expectation_values_from_parities(parities)

        self.assertEqual(len(expectation_values.values), 3)
        self.assertAlmostEqual(expectation_values.values[0], -0.47058823529411764)
        self.assertAlmostEqual(expectation_values.values[1], 0.030042918454935622)
        self.assertAlmostEqual(expectation_values.values[2], 0.48514851485148514)

        self.assertEqual(len(expectation_values.covariances), 3)
        self.assertTrue(np.allclose(expectation_values.covariances[0], np.array([[0.014705882352941176]])))
        self.assertTrue(np.allclose(expectation_values.covariances[1], np.array([[0.00428797]])))

        self.assertTrue(np.allclose(expectation_values.covariances[2], np.array([[0.0075706]])))

    def test_expectation_values_to_real(self):
        # Given
        expectation_values = ExpectationValues(np.array([0.+0.1j, 0.+1e-10j, -1.0]))
        target_expectation_values = ExpectationValues(np.array([0., 0., -1.0]))
        
        # When
        real_expectation_values = expectation_values_to_real(expectation_values)

        # Then
        np.testing.assert_array_equal(real_expectation_values.values, target_expectation_values.values)
    
    def test_convert_bitstring_to_int(self):
        bitstring = (0, 1, 0, 1, 0, 1)
        self.assertEqual(convert_bitstring_to_int(bitstring), 42)
