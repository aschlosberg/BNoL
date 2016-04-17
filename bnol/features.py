import numpy as np
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

        self.__matrixA(discrete, classes)

    def __getSpecimenPairs(self, classes):
        grouped = [np.nonzero(classes==g)[0] for g in [False, True]]
        return list(itertools.product(*grouped))

    def __matrixA(self, discrete, classes):
        pairs = self.__getSpecimenPairs(classes)
        A = np.empty((len(pairs), discrete.shape[1]), dtype='bool')

        for i, (d1, d2) in enumerate(pairs):
            A[i,:] = np.logical_xor(discrete[d1,:], discrete[d2,:])

        return A
