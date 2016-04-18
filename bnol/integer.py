"""Integer-programming methods for finding reduced feature sets following discretization."""

import numpy as np, numpy.ma as ma
from . import information, utility
import itertools

class MinFeatureSet(object):
    """Selection of the minimum feature set required to discriminate between specimen classes.

    Based upon methods described in `Selection of discriminative genes in microarray experiments using mathematical programming <http://test.acs.org.au/__data/assets/pdf_file/0018/15381/JRPIT39.4.287.pdf>`_
    """
    def __init__(self, specimens, classes, specimensAreDiscrete=False):

        # Technically we do not need to have the user specify if the specimens are discrete as this is included in the dtype of the array.
        # However it is best that they are specific about what they are providing such that they understand the processing that will occur.
        if specimensAreDiscrete:
            if not specimens.dtype=='bool':
                raise Exception("Non-discrete specimens provided when specimensAreDiscrete flag set to True. Expecting numpy.darray with dtype of 'bool'.")
            discrete = np.copy(specimens)
            includeFeatures = np.ones(specimens.shape[1]).astype('bool')
        else:
            if specimens.dtype=='bool':
                raise Exception("Discrete specimens provided when specimensAreDiscrete flag set to False. Expecting numpy.darray with dtype of either 'float' or 'init'.")
            discretize = information.Discretize()
            discrete = discretize.fit_transform(specimens, classes)
            includeFeatures = discretize.includeFeatures

        assert discrete.shape[0]==specimens.shape[0], "Number of specimens changed when discretized."

        if not discrete.shape[0]==len(classes):
            raise Exception("Number of specimens does not match number of classes for minmum feature-set selection.")

        self._discrete = discrete
        self._classes = classes
        self._reduce()

    def _reduce(self):
        rawA = self._matrixA(self._discrete, self._classes)
        A = ma.masked_array(rawA, np.zeros(rawA.shape))

        # R0
        featureCoveragePerPair = self._featureCoveragePerPair(A)
        if np.any(featureCoveragePerPair==0):
            raise InfeasibleInstanceException("Feature reduction infeasible by this method. See Berretta et al (2007); can not reduce discrete feature set as Min FEATURE Reduction R0 was not met.")

        self._R1(A)

    def _featureCoveragePerPair(self, A):
        coverage = np.sum(A, axis=1)
        assert A.shape[0]==len(coverage), "Matrix A summed across incorrect axis for determining feature coverage per pair"
        return coverage

    def _R1(self, A):
        featureCoveragePerPair = self._featureCoveragePerPair(A)
        indicesOfSingleFeaturePairs = np.nonzero(featureCoveragePerPair==1)[0]
        valuesOfSingleFeaturePairs = A[indicesOfSingleFeaturePairs,:]
        doesEachFeatureCoverSingletons = np.any(valuesOfSingleFeaturePairs, axis=0)
        assert doesEachFeatureCoverSingletons.shape==(len(self._classes),), "Features covering singly-covered pairs were calculated across the wrong axis for Min FEATURE Reduction R1"
        indicesOfFeaturesCoveringSingletons = np.nonzero(doesEachFeatureCoverSingletons)[0]

        if len(indicesOfFeaturesCoveringSingletons):
            canEachPairBeDeleted = np.any(A[:,indicesOfFeaturesCoveringSingletons], axis=1)
            assert canEachPairBeDeleted.shape==(A.shape[0],), "Pairs for deletion due to Min FEATURE Reduction R1 were calculated across the wrong axis"
            A.mask[canEachPairBeDeleted,:] = True
        else:
            canEachPairBeDeleted = ma.masked_array(np.zeros(A.mask.shape[0], dtype='bool'), np.any(A.mask, axis=1))

        return canEachPairBeDeleted

    def _classGroups(self, classes):
        return [np.nonzero(classes==g)[0] for g in [False, True]]

    def _getSpecimenPairsA(self, classes):
        grouped = self._classGroups(classes)
        return list(itertools.product(*grouped))

    def _matrix(self, discrete, pairs, logical):
        matrix = np.empty((len(pairs), discrete.shape[1]), dtype='bool')
        for i, (d1, d2) in enumerate(pairs):
            matrix[i,:] = logical(discrete[d1,:], discrete[d2,:])
        return matrix

    def _matrixA(self, discrete, classes):
        pairs = self._getSpecimenPairsA(classes)
        return self._matrix(discrete, pairs, np.logical_xor)

class AlphaBetaK(MinFeatureSet):
    def _getSpecimenPairsB(self, classes):
        grouped = self._classGroups(classes)
        allPairs = np.empty((0,2))
        for g in grouped:
            groupPairs = list(map(list, itertools.combinations(g, 2)))
            allPairs = np.vstack((allPairs, groupPairs))
        return allPairs.astype('int')

    def _matrixB(self, discrete, classes):
        pairs = self._getSpecimenPairsB(classes)
        return self._matrix(discrete, pairs, np.equal)

class InfeasibleInstanceException(Exception):
    pass
