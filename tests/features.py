import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from bnol import features, data
import unittest
import numpy as np

class FeaturesTester(unittest.TestCase):

    def setUp(self):
        (self.discrete, self.classes) = data.BerrettaDiscrete()

        featureSet = features.AlphaBetaK(self.discrete, self.classes, specimensAreDiscrete=True)
        self.pairs = {}
        self.matrices = {}
        for v in ['A', 'B']:
            self.pairs[v] = getattr(featureSet, "_getSpecimenPairs%s" % v)(self.classes)
            self.matrices[v] = getattr(featureSet, "_matrix%s" % v)(self.discrete, self.classes)

    def tearDown(self):
        for k in ['discrete', 'classes', 'pairs', 'matrices']:
            delattr(self, k)

    def test_min_feature_set_selection(self):
        pass

    def __test_pair_generation(self, AorB, expected):
        self.assertTrue(np.alltrue(self.pairs[AorB]==expected), "Specimen-pair generation (%s) for does not match the demonstrative pairings in Berretta et al 2007." % AorB)

    def test_min_feature_set_pair_generation_A(self):
        expected = np.asarray([
            [0,2],
            [0,3],
            [0,4],
            [1,2],
            [1,3],
            [1,4]
        ])
        self.__test_pair_generation('A', expected)

    def test_alpha_beta_K_pair_generation_B(self):
        expected = np.asarray([
            [0,1],
            [2,3],
            [2,4],
            [3,4]
        ])
        self.__test_pair_generation('B', expected)

    def __test_matrix_calculation(self, AorB, expected):
        matrix = self.matrices[AorB]
        self.assertEqual(matrix.shape[0], len(self.pairs[AorB]), "Matrix %s should have same number of rows as pairs of different-class features" % AorB)
        self.assertEqual(matrix.shape[1], self.discrete.shape[1], "Matrix %s should have same number of features as does the first parameter passed (discrete features)" % AorB)
        self.assertTrue(np.alltrue(matrix==expected), "Matrix %s does not match the demonstrative values in Berretta et al 2007." % AorB)

    def test_min_feature_set_matrix_A_calculation(self):
        expected = np.asarray([
            [0,0,0,0,1],
            [0,1,1,1,0],
            [1,1,0,1,0],
            [1,1,1,0,1],
            [1,0,0,1,0],
            [0,0,1,1,0],
        ], dtype='bool')
        self.__test_matrix_calculation('A', expected)

    def test_alpha_beta_K_matrix_B_calculation(self):
        expected = np.asarray([
            [0,0,0,1,1],
            [1,0,0,0,0],
            [0,0,1,0,0],
            [0,1,0,1,1],
        ], dtype='bool')
        self.__test_matrix_calculation('B', expected)

if __name__=='__main__':
    unittest.main()
