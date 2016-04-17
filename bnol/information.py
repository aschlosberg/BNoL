import numpy as np
from scipy.stats import entropy
from . import utility

def Entropy(distributions):
    """Calculate information entropy (in bits) for a set of distributions.
    Wrapper for scipy.stats.entropy(distributions.T) with base=2.

    Args:
        distributions (numpy.darray): shape (n,p) representing relative frequencies of the p features across n specimens.

    Returns:
        numpy.darray(): n information entropy values, one for each specimen.
    """

    return entropy(np.asarray(distributions).T, base=2)

class Divergence(object):
    """Calculate different divergence measures, D(P||Q), to describe the 'statistical distance' between a set of probability distributions and a reference.

    Utilises scipy.stats.entropy() under the hood but can accept multiple values for Q simultaneously.

    See `Wikipedia: Statistical distance <https://en.wikipedia.org/wiki/Statistical_distance>`_ as well as links in individual measures.

    Args:
        P (Optional[numpy.darray]): shape (1,p) defining a reference sequence against which divergence is calculated. If not provided then a p-dimensional discrete uniform is used.
        Qs (numpy.darray): shape (n,p) defining the relative frequencies of features in the n specimens. NOTE: Although it has a default value of None, this is not optional (the ordering of P and Qs parameters is a sane choice given scipy.stats.entropy and this requires that Qs have a default if P does).

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
            numpy.darray: n divergence values, one for each specimen.
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
            numpy.darray: n divergence values, one for each specimen.
        """
        Ms = (self.Qs + self.P) / 2
        D_PM = Divergence(self.P, Ms).KL()
        D_QM = [Divergence(self.Qs[i], Ms[i]).KL()[0] for i in range(self.nQs)]
        return (D_PM + D_QM) / 2

def Complexity(distributions, reference=None):
    """For each of a group of n specimens, return the specimen entropy multiplied by the Jensen-Shannon divergence (each in bits).

    See `Berretta et al. Cancer biomarker discovery: The entropic hallmark <https://dx.doi.org/10.1371/journal.pone.0012262>`_.

    Args:
        distributions (numpy.darray): shape (n,p) defining the relative frequencies of features in the n specimens.
        reference (Optional[numpy.darray]): shape(1,p) defining a reference sequence against which complexity is calculated. If not provided then the average of all distributions is used.

    Returns:
        numpy.darray: n complexity values, one for each specimen.
    """
    if reference is None:
        reference = np.sum(distributions, axis=0)
        assert len(reference)==distributions.shape[1], "Calculated average reference distribution across wrong axis"

    E = Entropy(distributions)
    D = Divergence(P=reference, Qs=distributions)
    return np.multiply(E, D.JS()) # element-wise multiplication

class Discretize:
    """Information-theoretic method for discrete feature selection based on Minimum Description Length Principle (MDLP).

    This method additionally provides an entropy-minimising means of defining a context-dependent threshold for over- / under-expression of genes between groups. This is the (binary) discretisation step.

    Along with gene-expression values, all specimens are labelled by class (i.e. context-dependency) such as cancer vs normal tissue, and a threshold is produced such that the entropy of classes within the groups (above and below said threshold) is minimised.

    Features are only included if the reduction in entropy, compared to no separation by a threshold, is sufficiently great (the MDLP criterion).

    See `Fayyad and Irani. Multi-interval discretization of continuous-valued attributes for classification learning <http://ourmine.googlecode.com/svn-history/r1513/trunk/share/pdf/fayyad93.pdf>`_.

    Attributes:
        baseEntropy (float): class entropy for the whole set of specimens.
        bestThresholds (numpy.darray): float; shape(n,) optimal, entropy-minimising, threshold for each of the n features.
        discretizedFeatures (numpy.darray): bool; shape (n,p); whether or not each specimen-feature value exceeds the optimal threshold for said feature.
        includeFeatures (numpy.darray): bool; shape(p,); whether or not the optimal threshold for each feature is sufficient such that the decrease in entropy meets the MDLP criterion.
    """
    def __init__(self):
        pass

    def fit(self, distributions, classes):
        """Determine threshold values for each feature.

        Args:
            distributions (numpy.darray): shape (n,p) defining the relative frequencies of features in the n specimens.
            classes (numpy.darray): shape (n,) defining which of the two classes each specimen belongs to; values will be cast with numpy.darray.astype('bool').

        Raises:
            Exception: if number of samples is different in distributions and classes arguments.
        """
        if not len(classes)==distributions.shape[0]:
            raise Exception("Number of class indicators for descritization must equal number of samples provided.")

        self.booleanClasses = classes.astype('bool')
        self.baseEntropy = self.__specimenClassEntropy(self.booleanClasses)
        self.distributions = distributions

        featureDecisions = map(self.__processFeature, range(distributions.shape[1])) # use a map so it can later be parallelised
        decisionTuples = zip(*featureDecisions)
        (self.includeFeatures, self.bestThresholds, self.discretizedFeatures) = map(lambda t: np.asarray(t).T, decisionTuples)

    def transform(self, distributions, allFeatures=False):
        """Not yet implemented.

        .. todo::
            Define a function to discretize novel specimens that weren't used in determining the optimal thresholds and feature selection.
        """
        pass

    def fit_transform(self, distributions, classes, allFeatures=False):
        """Determine threshold values as if only calling fit(distributions, classes) and return the discretized features.

        .. note::
            Use Discretize.includeFeatures attribute to determine which features are included when allFeatures==False.

        Args:
            distributions (numpy.darray): shape (n,p) defining the relative frequencies of features in the n specimens.
            classes (numpy.darray): shape (n,) defining which of the two classes each specimen belongs to; values will be cast with numpy.darray.astype('bool').
            allFeatures (bool): if False will exclude those features for which the MDLP criterion was not met.

        Returns:
            numpy.darray: boolean; shape (n,p) if allFeatures==true otherwise shape (n`,p) where n` is the number of features for which MDLP criterion was met; boolean value represents whether or not each specimen-feature value exceeds the optimal threshold for said feature.
        """
        self.fit(distributions, classes)
        return self.discretizedFeatures if allFeatures else self.discretizedFeatures[:,self.includeFeatures]

    def __processFeature(self, i):
        """Determine threshold value for a single feature and decide if the MDLP criterion is met.

        Args:
            i (int): index of the particular feature.

        Returns:
            tuple(bool, float, numpy.darray): [0] for the best threshold, has the MDLP criterion been met (i.e should we include this feature); [1] best threshold; [2] for each specimen, does it exceed the best threshold.
        """

        feature = self.distributions[:,i]
        nSamples = len(self.booleanClasses)
        minEntropy = self.baseEntropy
        bestThreshold = None
        bestSeparationEntropies = None

        for threshold in sorted(feature)[:-1]: # therefore MUST use > in splitting and not >=
            """
            This is worst-case O(n) but I think there may be an O(log(n)) solution where n is the number of specimens.
            The problem with a binary search, however, is that we don't know which way to go to decrease entropy.
            """
            separation = self.__getSeparation(feature, threshold)

            n1 = np.count_nonzero(separation)
            Ent1 = self.__specimenClassEntropy(self.booleanClasses[separation])

            n2 = nSamples - n1
            Ent2 = self.__specimenClassEntropy(self.booleanClasses[~separation])

            thresholdEntropy = (n1 * Ent1 + n2 * Ent2) / nSamples

            if thresholdEntropy<minEntropy:
                minEntropy = thresholdEntropy
                bestThreshold = threshold
                bestSeparationEntropies = [Ent1, Ent2]
            else:
                break

        bestSeparation = self.__getSeparation(feature, bestThreshold)
        delta = self.__deltaMDLP(i, feature, bestSeparation, bestSeparationEntropies) # see Fayyad and Irani paper
        mdlpCriterion = (np.log2(nSamples-1) + delta) / nSamples
        gain = self.baseEntropy- minEntropy

        return (gain>mdlpCriterion, bestThreshold, bestSeparation)

    def __deltaMDLP(self, i, feature, bestSeparation, bestSeparationEntropies):
        """Calculate the delta value for a particular feature, as defined in Fayyad and Irani paper for calculating MDLP criterion."""
        assert bestSeparation.dtype=='bool', "Expecting boolean class values when calculating delta value."
        assert len(bestSeparationEntropies)==2, "Only two separations can be used for calculating MDLP criterion."

        classCounts = list(map(lambda s: len(np.unique(self.booleanClasses[s])), [bestSeparation, ~bestSeparation]))
        return np.log2(6) - (2*self.baseEntropy - np.dot(classCounts, bestSeparationEntropies))

    def __getSeparation(self, feature, threshold):
        """Ensure that thresholding is performed identically at all times. Threshold candidates are simply the values of the features so we MUST use > and not >=.

        Args:
            feature (numpy.darray): shape (n,) vector defining relative frequency of feature across all specimens.
            threshold (float): value for separation of specimens into groups based on their feature value.

        Returns:
            numpy.darray: boolean array defining if a specimen is greater than the threshold.
        """
        return feature<=threshold

    def __specimenClassEntropy(self, booleanClasses):
        """Calculate the entropy of classes a group of specimens. Note that this may be the full set of features or separations based on a threshold."""
        assert booleanClasses.dtype=='bool', "Expecting boolean class values when calculating entropy."
        frequencies = np.zeros(2)
        frequencies[0] = np.count_nonzero(booleanClasses)
        frequencies[1] = len(booleanClasses) - frequencies[0]
        return Entropy(frequencies)
