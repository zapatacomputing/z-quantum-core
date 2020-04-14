import unittest
import os
import random
import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group

from openfermion.utils import qubit_operator_sparse
from .utils import (convert_array_to_dict, convert_dict_to_array,
                    dec2bin, bin2dec, 
                    is_identity, is_unitary, compare_unitary, RNDSEED,
                    ValueEstimate, save_value_estimate, load_value_estimate, 
                    save_list, load_list, create_object)

class TestUtils(unittest.TestCase):

    def test_real_array_conversion(self):
        arr = np.array([[1., 2.], [3., 4.]])
        dictionary = convert_array_to_dict(arr)
        new_arr = convert_dict_to_array(dictionary)
        self.assertTrue(np.allclose(arr, new_arr))

    def test_complex_array_conversion(self):
        arr = np.array([[1., 2.], [3., 4.j]])
        dictionary = convert_array_to_dict(arr)
        new_arr = convert_dict_to_array(dictionary)
        self.assertTrue(np.allclose(arr, new_arr))

    def test_dec_bin_conversion(self):
        integer = random.randint(1,10**9)
        integer2 = bin2dec(dec2bin(integer,30))
        self.assertEqual(integer, integer2)

    def test_is_identity(self):
        # Given
        matrix1 = np.eye(4)
        random.seed(RNDSEED)
        matrix2 = np.array([[random.uniform(-1,1) for x in range(0,4)] for y in range(0,4)])
        # When/Then
        self.assertTrue(is_identity(matrix1))
        self.assertFalse(is_identity(matrix2))
    
    def test_is_unitary(self):
        # Given
        U = unitary_group.rvs(4)
        # When/Then
        self.assertTrue(is_unitary(U))

    def test_compare_unitary(self):
        # Given
        U1 = unitary_group.rvs(4)
        U2 = unitary_group.rvs(4)
        # When/Then
        self.assertFalse(compare_unitary(U1,U2))

    def test_sample_from_probability_distribution(self):
        # TODO
        pass

    def test_convert_bitstrings_to_tuples(self):
        # TODO
        pass

    def test_convert_tuples_to_bitstrings(self):
        # TODO
        pass

    def test_value_estimate_dict_conversion(self):
        # TODO
        pass

    def test_value_estimate_io(self):
        value = -1.0
        precision = 0.1
        
        value_estimate_object = ValueEstimate(value, precision)

        save_value_estimate(value_estimate_object, 'value_estimate.json')
        value_estimate_object_loaded = load_value_estimate('value_estimate.json')

        self.assertEqual(value_estimate_object.value,
                         value_estimate_object_loaded.value)
        self.assertEqual(value_estimate_object.precision,
                         value_estimate_object_loaded.precision)
        
        value_estimate_object = ValueEstimate(value)

        save_value_estimate(value_estimate_object, 'value_estimate.json')
        value_estimate_object_loaded = load_value_estimate('value_estimate.json')

        self.assertEqual(value_estimate_object.value,
                         value_estimate_object_loaded.value)
        self.assertEqual(value_estimate_object.precision,
                         value_estimate_object_loaded.precision)
        
        os.remove('value_estimate.json')

    def test_list_io(self):
        l = [0.1, 0.3, -0.3]
        save_list(l, 'list.json')
        loaded_l = load_list('list.json')
        self.assertListEqual(l, loaded_l)
        os.remove('list.json')
    
    # def test_create_object(self):
        # TODO: not sure how to test this
        # # Given
        # specs = {}
        # specs['module_name'] = '.interfaces.mock_objects'
        # specs['function_name'] = 'MockOptimizer'

        # # When
        # mock_optimizer = create_object(specs)
        
        # # Then