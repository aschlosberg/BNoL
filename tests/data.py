import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from bnol import utility, data
import unittest
import numpy as np

class DataTester(unittest.TestCase):

    def test_sample_data_from_Berretta_2010_cancer_entropy_paper(self):
        freqs = data.BerrettaExpression()

        self.assertEqual(freqs.shape, (4,5), "Berretta data should have 4 specimens with 5 features in each")

        normalizedSample = utility.Normalize(freqs[-1])
        sameSizeUniform = utility.DiscreteUniform(freqs.shape[1])
        self.assertTrue(np.alltrue(normalizedSample==sameSizeUniform), "Final specimen in Berretta data should be uniform distribution")

        first = sorted(freqs[0])
        second = sorted(freqs[1])
        self.assertTrue(np.alltrue(first==second), "First and second specimens in Berretta data should be permutations")

    def test_sample_data_from_Berretta_2007_feature_set_paper(self):
        (features, classes) = data.BerrettaDiscrete()

        self.assertEqual(features.shape, (5,5), "Berretta discrete data should have 5 specimens with 5 features in each")
        self.assertEqual(classes.shape, (5,), "Berretta discrete data classes should have 5 specimens")
        self.assertEqual(features.dtype, 'bool', "Berretta discrete data should be boolean")
        self.assertEqual(classes.dtype, 'bool', "Berretta discrete data classes should be boolean")

if __name__=='__main__':
    unittest.main()
