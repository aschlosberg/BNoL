import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from bnol import utility, data
import unittest
import numpy as np

class DataTester(unittest.TestCase):

    def test_sample_data_from_Berretta_2010_paper(self):
        freqs = data.BerrettaExpression()

        self.assertEqual(freqs.shape, (4,5), "Berretta data should have 4 specimens with 5 features in each")

        normalizedSample = utility.Normalize(freqs[-1])
        sameSizeUniform = utility.DiscreteUniform(freqs.shape[1])
        self.assertTrue(np.alltrue(normalizedSample==sameSizeUniform), "Final specimen in Berretta data should be uniform distribution")

        first = sorted(freqs[0])
        second = sorted(freqs[1])
        self.assertTrue(np.alltrue(first==second), "First and second specimens in Berretta data should be permutations")

if __name__=='__main__':
    unittest.main()
