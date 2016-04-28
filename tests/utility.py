import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from bnol import utility, data
import unittest
from numpyfied import NumpyfiedTestCase
import numpy as np

class UtilityTester(NumpyfiedTestCase):

    def test_probability_distribution_normalizer(self):
        freqs = data.BerrettaExpression()
        normalized = utility.Normalize(freqs)
        summed = np.sum(normalized, axis=1)

        self.assertEqual(freqs.shape, normalized.shape, "Probability normalization modifies shape of data")
        self.assertEqual(len(summed), normalized.shape[0], "Summing normalized probability distributions over wrong axis")
        self.assertAllTrue(summed==1, "Normalized probability distributions do not add to one")

    def test_discrete_uniform_distribution_generator(self):
        for n in range(1,25):
            distribution = utility.DiscreteUniform(n)
            uniqueValues = np.unique(distribution)

            self.assertEqual(distribution.shape, (1,n), "Expected %d values in discrete uniform distribution" % n)
            self.assertEqual(len(uniqueValues), 1, "All values in discrete uniform distribution must be equal for n=%d" % n)
            self.assertIsClose(np.sum(distribution), 1, "Uniform distribution not normalized; values do not sum to one for n=%d" %n)

    def test_expand_vector_to_two_dimensions(self):
        np.random.seed(42)
        oneDimensional = np.random.random_sample((10,))
        twoDimensional = utility.VectorToMatrix(oneDimensional)

        self.assertEqual(oneDimensional.shape, (10,), "One-dimensional vector does not have correct number of dimensions")
        self.assertEqual(twoDimensional.shape, (1,10), "Two-dimensional vector not correctly expanded")
        self.assertAllTrue(oneDimensional==twoDimensional[0], "Values of two-dimensional vector do not match originals")

    def test_do_not_expand_matrix_if_thought_to_be_vector(self):
        np.random.seed(42)
        matrix = np.random.random_sample((10, 20))
        matrixCopy = utility.VectorToMatrix(matrix)

        self.assertEqual(matrix.shape, matrixCopy.shape, "Shape of matrix changed")
        self.assertAllTrue(matrix==matrixCopy, "Values of matrix do not match originals")
        self.assertTrue(matrix is matrixCopy, "Copy of matrix made when expecting identical object to be returned")

    def test_boolean_list_indexing(self):
        values = list(range(5))
        vectors = [
            ([True, True, False, False, False], [0,1]),
            ([False, True, False, True, False], [1,3]),
            ([True]*5, values[:]),
            ([False]*5, []),
            ([0,1,2,3,4], [1,2,3,4]),
        ]

        for i, (included, output) in enumerate(vectors):
            sublist = utility.BooleanListIndexing(values, included)
            self.assertEqual(len(sublist), len(output), "Boolean list indexing returned list of incorrect length for test vector %d" % i)
            self.assertAllTrue(sublist==output, "Boolean list indexing returned incorrect values for test vector %d" % i)

if __name__=='__main__':
    unittest.main()
