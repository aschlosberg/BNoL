import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from bnol import utility, data
import unittest
import numpy as np

class UtilityTester(unittest.TestCase):

    def test_probability_distribution_normalizer(self):
        freqs = data.BerrettaExpression()
        normalized = utility.Normalize(freqs)
        summed = np.sum(normalized, axis=1)

        self.assertEqual(freqs.shape, normalized.shape, "Probability normalization modifies shape of data")
        self.assertEqual(len(summed), normalized.shape[0], "Summing normalized probability distributions over wrong axis")
        self.assertTrue(np.alltrue(summed==1), "Normalized probability distributions do not add to one")

    def test_discrete_uniform_distribution_generator(self):
        for n in range(1,25):
            distribution = utility.DiscreteUniform(n)
            uniqueValues = np.unique(distribution)

            self.assertEqual(distribution.shape, (1,n), "Expected %d values in discrete uniform distribution" % n)
            self.assertEqual(len(uniqueValues), 1, "All values in discrete uniform distribution must be equal for n=%d" % n)
            self.assertTrue(np.isclose(np.sum(distribution), 1, data.epsilon()), "Uniform distribution not normalized; values do not sum to one for n=%d" %n)

    def test_expand_vector_to_two_dimensions(self):
        oneDimensional = np.empty((10,))
        twoDimensional = utility.VectorToMatrix(oneDimensional)

        self.assertEqual(oneDimensional.shape, (10,), "One-dimensional vector does not have correct number of dimensions")
        self.assertEqual(twoDimensional.shape, (1,10), "Two-dimensional vector not correctly expanded")
        self.assertTrue(np.alltrue(oneDimensional==twoDimensional[0]), "Values of two-dimensional vector do not match originals")

    def test_do_not_expand_matrix_if_thought_to_be_vector(self):
        matrix = np.empty((10, 20))
        matrixCopy = utility.VectorToMatrix(matrix)

        self.assertEqual(matrix.shape, matrixCopy.shape, "Shape of matrix changed")
        self.assertTrue(np.alltrue(matrix==matrixCopy), "Values of matrix do not match originals")
        self.assertTrue(matrix is matrixCopy, "Copy of matrix made when expecting identical object to be returned")

if __name__=='__main__':
    unittest.main()