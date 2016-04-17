import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from bnol import features, data
import unittest
import numpy as np

class FeaturesTester(unittest.TestCase):

    def test_min_feature_set_selection(self):
        (specimens, classes) = data.BerrettaDiscrete()

        featureSet = features.MinFeatureSet(specimens, classes, specimensAreDiscrete=True)

    def test_min_feature_set_pair_generation(self):
        (__, classes) = data.BerrettaDiscrete()

        mock = MinFeatureSet() # note that this is NOT features.MinFeatureSet as it is extended to provide access to private methods; must have the same class name to access private methods
        pairs = mock.getSpecimenPairs(classes)
        expected = np.asarray([
            [0,2],
            [0,3],
            [0,4],
            [1,2],
            [1,3],
            [1,4]
        ])

        self.assertTrue(np.alltrue(pairs==expected), "Specimen-pair generation for min feature set does not match the demonstrative pairings in Berretta et al 2007.")

    def test_min_feature_set_matrix_A_calculation(self):
        (discrete, classes) = data.BerrettaDiscrete()

        mock = MinFeatureSet() # see note in test_min_feature_set_pair_generation
        pairs = mock.getSpecimenPairs(classes)
        A = mock.matrixA(discrete, classes)

        expected = np.asarray([
            [0,0,0,0,1],
            [0,1,1,1,0],
            [1,1,0,1,0],
            [1,1,1,0,1],
            [1,0,0,1,0],
            [0,0,1,1,0],
        ], dtype='bool')

        self.assertEqual(A.shape[0], len(pairs), "Matrix A for min feature set problem should have same number of rows as pairs of different-class features")
        self.assertEqual(A.shape[1], discrete.shape[1], "Matrix A for min feature set problem should have same number of features as does the first parameter passed (discrete features)")
        self.assertTrue(np.alltrue(A==expected), "Matrix A for min feature set problem does not match the demonstrative values in Berretta et al 2007.")

class MinFeatureSet(features.MinFeatureSet):
    """Provide access to private methods that would benefit from testing because we have well-described expected results."""
    def __init__(self):
        pass

    def getSpecimenPairs(self, classes):
        return self.__getSpecimenPairs(classes)

    def matrixA(self, discrete, classes):
        return self.__matrixA(discrete, classes)

if __name__=='__main__':
    unittest.main()
