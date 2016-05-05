import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import unittest
from numpyfied import NumpyfiedTestCase
import numpy as np
from bnol import distance

class DistanceTester(NumpyfiedTestCase):

    def setUp(self):
        self.samples = np.asarray([
            [1,2,3],
            [1,2,3], # identical to first
            [1,0,0],
            [0,1,0], # orthogonal to previous
        ])
        self.nSamples = 4

    def _common_distance_matrix_tests(self, fn):
        distances = getattr(distance, fn)(self.samples)
        self.assertEqual(len(distances.shape), 2, "Distance matrices should have 2 dimensions (%s)" % fn)
        self.assertEqual(distances.shape[0], distances.shape[1], "Distance matrices should be square (%s)" % fn)
        self.assertEqual(distances.shape[0], self.nSamples, "Distance matrices should be pair-wise, and thus have same dimension length as number of samples (%s)" % fn)
        self.assertAllTrue(np.diag(distances)==0, "Sample distance from itself must be zero; non-zero value on diagonal (%s)" % fn)
        self.assertEqual(distances[0,1], distances[1,0], "Identically-valued samples should have zero distance (%s)" % fn)
        self.assertAllTrue(distances==distances.T, "Distance matrices should be symmetrical; transpose is not equal to original (%s)" % fn)
        return distances

    def test_cosine_distance(self):
        distances = self._common_distance_matrix_tests("Cosine")
        self.assertEqual(distances[2,3], 1, "Cosine distance between orthogonal vectors should be one")

    def test_jensen_shannon_divergence_distance(self):
        distances = self._common_distance_matrix_tests("JensenShannon")

if __name__=='__main__':
    unittest.main()
