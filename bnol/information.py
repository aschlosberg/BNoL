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

        Example:
            JSD is calculated from KLD as follows::

                let KLD = D(P||Q);
                let M = (P+Q) / 2;
                JSD(P||Q) = (D(P||M) + D(Q||M)) / 2

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
