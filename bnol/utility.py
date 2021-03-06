"""General helper functions used throughout modules but likely of use in other settings."""

import numpy as np

def Normalize(freqs):
    """Probability-normalize features for a set of specimens (i.e. divide each by their total such that they sum to one).

    Args:
        freqs (numpy.ndarray): shape (n,p) defining relative frequencies of p features in each of n specimens.

    Returns:
        numpy.ndarray: shape (n,p) where the sum across axis=1 is one for all n specimens.
    """
    freqs = VectorToMatrix(np.asarray(freqs))
    inverse = np.diag(1 / np.sum(freqs, axis=1))
    return np.dot(inverse, freqs)

def DiscreteUniform(p):
    """Generate a numpy array of equal-value floats representing a discrete uniform distribution.

    Args:
        p (int): the number of possible outcomes for the distribution; values are cast as int(p).

    Returns:
        numpy.ndarray: shape (1,p) where all values are equal to 1/p.

    Raises:
        Exception: if p, after being cast to an integer, is less than one.
    """
    p = int(p)
    if p<1:
        raise Exception("Can only generate discrete uniform distribution for positive integers")
    return np.ones((1,p))/p;

def VectorToMatrix(vec):
    """Most BNoL functions are developed to work on matrices such that each row represents a specimen. In some cases we may wish to pass a single specimen that is defined as a one-dimensional numpy.ndarray and this will cause problems if darray.shape[1] is utilised.

    Expand dimensions for such vectors such that they have shape (1,p) instead of (p,). Leave matrices unchanged.

    Args:
        vec (numpy.ndarray): either a (n,p) matrix defining n specimens or a (p,) vector defining a single specimen.

    Returns:
        numpy.ndarray: the original object if it was an (n,p) matrix or an expanded-dimension (1,p) vector of the single-specimen values.
    """
    return np.expand_dims(vec, axis=0) if len(vec.shape)==1 else vec

def BooleanListIndexing(listVals, included=None):
    """Mimic numpy indexing by boolean flags, i.e. return those values of listVals for which the corresponding value in included is truthy.

    If include is None then return the original list.

    Args:
        listVals (list): values from which we wish to return a subset.
        included (list): whether or not to return the corresponding value in listVals.

    Returns:
        list: numpy equivalent of listVals[included==True]

    Raises:
        Exception: if lengths of listVals and include differ.
    """
    if included is None:
        return listVals
    else:
        if not len(listVals)==len(included):
            raise Exception("Boolean list indexing requires identical-length lists as arguments.")
        return [listVals[idx] for (idx, incl) in enumerate(included) if incl]
