import numpy as np

def BerrettaExpression():
    """Example frequency distributions with desirable properties:

    - First two specimens are permutations of each other;
    - Final specimen is uniformly distributed.

    See `Berretta et al. Cancer biomarker discovery: The entropic hallmark <https://dx.doi.org/10.1371/journal.pone.0012262>`_.

    Returns:
        numpy.ndarray: shape (4,5)
    """
    return np.asarray([[4,3,2,1,0.1],[0.1,1,2,3,4],[5,2,5,1,3],[2,2,2,2,2]])

def epsilon():
    """Epsilon value for use in testing as the tolerance in numpy.isclose().

    Returns:
        float: 1e-12.
    """
    return 1e-12
