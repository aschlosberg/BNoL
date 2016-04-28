"""Information-theoretic measures (e.g. entropy and divergence) and analyses (e.g. Minimum Description Length Principle for feature selection)."""

import numpy as np
from scipy.stats import entropy
from . import utility
import multiprocessing
import logging

#logger = multiprocessing.log_to_stderr()
#logger.setLevel(multiprocessing.SUBDEBUG)

def Entropy(distributions):
    """Calculate information entropy (in bits) for a set of distributions.
    Wrapper for scipy.stats.entropy(distributions.T) with base=2.

    Args:
        distributions (numpy.ndarray): shape (n,p) representing relative frequencies of the p features across n specimens.

    Returns:
        numpy.ndarray(): n information entropy values, one for each specimen.
    """

    return entropy(np.asarray(distributions).T, base=2)

class Divergence(object):
    """Calculate different divergence measures, D(P||Q), to describe the 'statistical distance' between a set of probability distributions and a reference.

    Utilises scipy.stats.entropy() under the hood but can accept multiple values for Q simultaneously.

    See `Wikipedia: Statistical distance <https://en.wikipedia.org/wiki/Statistical_distance>`_ as well as links in individual measures.

    Args:
        P (Optional[numpy.ndarray]): shape (1,p) defining a reference sequence against which divergence is calculated. If not provided then a p-dimensional discrete uniform is used.
        Qs (numpy.ndarray): shape (n,p) defining the relative frequencies of features in the n specimens. NOTE: Although it has a default value of None, this is not optional (the ordering of P and Qs parameters is a sane choice given scipy.stats.entropy and this requires that Qs have a default if P does).

    Raises:
        Exception: If no value is provided for Qs.
        Exception: If P is provided and it has a different number of features to Qs (i.e. not P.shape[1]==Qs.shape[1]).
        Exception: If P is provided and it contains more than one reference (i.e. P.shape[0]>1).
    """

    def __init__(self, P=None, Qs=None):
        """Perform sanity checks on the sizes of the distributions of P and Q."""
        if Qs is None:
            raise Exception("No default value is possible for Qs when calculating divergence")

        Qs = utility.VectorToMatrix(Qs)

        if P is None:
            P = utility.DiscreteUniform(Qs.shape[1])
        else:
            P = utility.VectorToMatrix(P)
            if not Qs.shape[1]==P.shape[1]:
                raise Exception("Can not calculate divergence for distributions of different sizes")
            if not P.shape[0]==1:
                raise Exception("Only one reference distribution should be provided for calculation of divergence")

        self.P = P
        self.Qs = Qs
        self.nQs = Qs.shape[0]

    def KL(self):
        """Return the Kullback-Leibler divergence (in bits).

        *Note that this is not symmetrical.*

        See `Wikipedia: Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_.

        Returns:
            numpy.ndarray: n divergence values, one for each specimen.
        """
        return np.asarray([entropy(self.P[0], self.Qs[i], base=2) for i in range(self.nQs)])

    def JS(self):
        """Return the Jensen-Shannon divergence (in bits).

        Unlike Kullback-Leibler divergence (KLD), this is symmetrical. The square-root of the value is also a metric so it will satisfy the triangle inequality.

        See `Wikipedia: Jensen-Shannon divergence <https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence>`_.

        .. note::
            JSD(P||Q) is calculated from KLD(P||Q) as follows::

                let M = (P+Q) / 2
                JSD(P||Q) = (KLD(P||M) + KLD(Q||M)) / 2

        Returns:
            numpy.ndarray: n divergence values, one for each specimen.
        """
        Ms = (self.Qs + self.P) / 2
        D_PM = Divergence(self.P, Ms).KL()
        D_QM = [Divergence(self.Qs[i], Ms[i]).KL()[0] for i in range(self.nQs)]
        return (D_PM + D_QM) / 2

def Complexity(distributions, reference=None):
    """For each of a group of n specimens, return the specimen entropy multiplied by the Jensen-Shannon divergence (each in bits).

    See `Berretta et al. Cancer biomarker discovery: The entropic hallmark <https://dx.doi.org/10.1371/journal.pone.0012262>`_.

    Args:
        distributions (numpy.ndarray): shape (n,p) defining the relative frequencies of features in the n specimens.
        reference (Optional[numpy.ndarray]): shape(1,p) defining a reference sequence against which complexity is calculated. If not provided then the average of all distributions is used.

    Returns:
        numpy.ndarray: n complexity values, one for each specimen.
    """
    if reference is None:
        reference = np.sum(distributions, axis=0)
        assert len(reference)==distributions.shape[1], "Calculated average reference distribution across wrong axis"

    E = Entropy(distributions)
    D = Divergence(P=reference, Qs=distributions)
    return np.multiply(E, D.JS()) # element-wise multiplication

def ParallelFeatureDiscretization(tupleArguments):
    """Determine threshold value for a single feature and decide if the MDLP criterion is met.

    This is not a method of class information.Discretize as we require a function defined at the top of the module to allow for pickling prior to use with multiprocessing.Pool.
    We are also limited to a single argument so the tuple is unwrapped as (feature, classes, baseEntropy).

    Args:
        tupleArguments (numpy.ndarray, numpy.ndarray, float): [0] feature: shape (n,1); feature values for all specimens; [1] classes: shape (n,) defining categorical designation of each specimen; [2] baseEntropy: entropy value pre-calculated for classes

    Returns:
        tuple(bool, float, numpy.ndarray, float, float): [0] for the best threshold, has the MDLP criterion been met (i.e should we include this feature); [1] best threshold; [2] for each specimen, does it exceed the best threshold; [3] entropy gain as defined in Fayyad and Irani paper; [4] MDLP criterion which entropy gain must exceed
    """
    (feature, classes, baseEntropy) = tupleArguments
    thresholds = sorted(np.unique(feature))
    nSamples = len(classes)

    # if all values are the same then it is impossible to improve entropy so bail out early with baseEntropy + 1 as the MDLP criterion as it is impossible to reach
    if len(thresholds)==1:
        return (False, thresholds[0], np.asarray([True]*nSamples), 0, baseEntropy + 1)

    minEntropy = baseEntropy
    bestThreshold = None
    bestSeparationEntropies = None

    for t, threshold in enumerate(thresholds[:-1]): # therefore MUST use > in splitting and not >=
        """
        This loop is O(n) if we consider its working to be constant. Can we find the optimal threshold faster? I thought that a binary partitioning O(logn) would work but the entropy is non-convex with respect to the threshold.
        The point of this function is to determine entropy-reducing ability of each feature so we can't make assumptions; thus I think O(n) is best possible as we must check every threshold.
        """
        separation = Discretize.getSeparation(feature, threshold) # boolean values indicating if each specimen surpasses the threshold for this feature
        negated = [separation, ~separation]
        n = [np.count_nonzero(s) for s in negated]
        ent = [Discretize.groupClassEntropy(classes[s]) for s in negated]
        thresholdEntropy = np.dot(n, ent) / nSamples

        if thresholdEntropy<minEntropy:
            minEntropy = thresholdEntropy
            bestThreshold = (thresholds[t] + thresholds[t+1])/2 # note that we only enumerate up to thresholds[-1]
            bestSeparationEntropies = ent[:]
        # used to have else: break here but I have demonstrated that the function is non-convex so this is wrong!

    bestSeparation = Discretize.getSeparation(feature, bestThreshold)
    delta = Discretize.deltaMDLP(classes, baseEntropy, bestSeparation, bestSeparationEntropies) # see Fayyad and Irani paper
    mdlpCriterion = (np.log2(nSamples-1) + delta) / nSamples
    gain = baseEntropy - minEntropy

    return (gain>mdlpCriterion, bestThreshold, bestSeparation, gain, mdlpCriterion)

class Discretize:
    """Information-theoretic method for discrete feature selection based on Minimum Description Length Principle (MDLP).

    This method additionally provides an entropy-minimising means of defining a context-dependent threshold for over- / under-expression of genes across categorical designations. This is the (binary) discretisation step.

    Along with gene-expression values, all specimens are labelled by class (i.e. context-dependency) such as cancer vs normal tissue, and a threshold is produced such that the entropy of classes within the groups (above and below said threshold) is minimised. Any number of classes (>=2) can be used.

    Features are only included if the reduction in entropy, compared to no separation by a threshold, is sufficiently great (the MDLP criterion).

    See `Fayyad and Irani. Multi-interval discretization of continuous-valued attributes for classification learning <http://ourmine.googlecode.com/svn-history/r1513/trunk/share/pdf/fayyad93.pdf>`_.

    Attributes:
        baseEntropy (float): class entropy for the whole set of specimens.
        bestThresholds (numpy.ndarray): float; shape(p,) optimal, entropy-minimising, threshold for each of the p features.
        discretizedFeatures (numpy.ndarray): bool; shape (n,p); whether or not each specimen-feature value exceeds the optimal threshold for said feature.
        includeFeatures (numpy.ndarray): bool; shape(p,); whether or not the optimal threshold for each feature is sufficient such that the decrease in entropy meets the MDLP criterion.
        gains (numpy.ndarray): float; shape(p,); entropy improvement based on best threshold for each feature.
        mdlpCriteria (numpy.ndarray): float; shape(p,); minimum gain required for inclusion of each feature.
    """
    def __init__(self):
        pass

    def fit(self, distributions, classes):
        """Determine threshold values for each feature.

        Args:
            distributions (numpy.ndarray): shape (n,p) defining the relative frequencies of features in the n specimens.
            classes (numpy.ndarray): shape (n,) defining categorical designation of each specimen.

        Raises:
            Exception: if number of samples is different in distributions and classes arguments.
        """
        if not len(classes)==distributions.shape[0]:
            raise Exception("Number of class indicators for descritization must equal number of samples provided.")

        self.classes = classes
        self.baseEntropy = self.groupClassEntropy(self.classes)
        self.distributions = distributions

        logging.info("Initializing multiprocessing pool for feature discretization")
        pool = multiprocessing.Pool()
        # pool.map() will only allow a single argument so combine them as a tuple
        mappingArguments = [(self.distributions[:,featureIdx], self.classes, self.baseEntropy) for featureIdx in range(distributions.shape[1])]
        logging.debug("Mapping feature-discretization function to %d features" % len(mappingArguments))
        featureDecisions = pool.map(ParallelFeatureDiscretization, mappingArguments)
        logging.debug("Parallel feature discretization completed")
        pool.close()

        decisionTuples = zip(*featureDecisions)
        (self.includeFeatures, self.bestThresholds, self.discretizedFeatures, self.gains, self.mdlpCriteria) = (np.asarray(t).T for t in decisionTuples)
        logging.info("Completed feature discretization")

    def transform(self, distributions, allFeatures=False):
        """Not yet implemented.

        .. todo::
            Define a function to discretize novel specimens that weren't used in determining the optimal thresholds and feature selection.
        """
        pass

    def fit_transform(self, distributions, classes, allFeatures=False):
        """Determine threshold values as if only calling fit(distributions, classes) and return the discretized features.

        .. tip::
            Use Discretize.includeFeatures attribute to determine which features are included when allFeatures==False.

        Args:
            distributions (numpy.ndarray): shape (n,p) defining the relative frequencies of features in the n specimens.
            classes (numpy.ndarray): shape (n,) defining categorical designation of each specimen.
            allFeatures (bool): if False will exclude those features for which the MDLP criterion was not met.

        Returns:
            numpy.ndarray: boolean; shape (n,p) if allFeatures==true otherwise shape (n`,p) where n` is the number of features for which MDLP criterion was met; boolean value represents whether or not each specimen-feature value exceeds the optimal threshold for said feature.
        """
        self.fit(distributions, classes)
        return self.discretizedFeatures if allFeatures else self.discretizedFeatures[:,self.includeFeatures]

    @staticmethod
    def deltaMDLP(classes, baseEntropy, bestSeparation, bestSeparationEntropies):
        """Calculate the delta value for a particular feature, as defined in Fayyad and Irani paper for calculating MDLP criterion."""
        assert bestSeparation.dtype=='bool', "Expecting boolean separation values when calculating delta value."
        assert len(bestSeparationEntropies)==2, "Only two separations can be used for calculating MDLP criterion."

        # Fayyad and Irani use k for number of classes prior to separation, and k1 / k2 for number of classes in above/below threshold separations
        k = len(np.unique(classes))
        k12 = [len(np.unique(classes[s])) for s in [bestSeparation, ~bestSeparation]]
        logging.debug("Calculating MDLP delta: %d classes prior to separation and %d / %d post." % (k, k12[0], k12[1]))
        for i in range(2):
            if k12[i]==1:
                assert bestSeparationEntropies[i]==0, "Separation with only one class must have zero entropy"
        return np.log2(3**k - 2) - (k*baseEntropy - np.dot(k12, bestSeparationEntropies))

    @staticmethod
    def getSeparation(feature, threshold):
        """Ensure that thresholding is performed identically at all times. Threshold candidates are simply the values of the features so we MUST use > and not >=.

        Args:
            feature (numpy.ndarray): shape (n,) vector defining relative frequency of feature across all specimens.
            threshold (float): value for separation of specimens into groups based on their feature value.

        Returns:
            numpy.ndarray: boolean array defining if a specimen is greater than the threshold.
        """
        return feature>threshold

    @staticmethod
    def groupClassEntropy(classes):
        """Calculate the entropy of classes a group of specimens. Note that this may be the full set of features or separations based on a threshold."""
        (__, freq) = np.unique(classes, return_counts=True)
        return Entropy(freq)
