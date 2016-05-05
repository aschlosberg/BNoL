import numpy as np
from scipy.spatial import distance
from math import sqrt
from . import information

def Generic(samples, fn):
    """Convenience wrapper for scipy.spacial.distance.squareform(scipy.spatial.distance.pdist(samples, fn))

    Args:
        samples (numpy.ndarray): shape (n,p); where n is the number of samples
        fn (str or function(sample, sample) => float): distance measure that returns a float for given samples

    Returns:
        numpy.ndarray: shape (n,p); symmetrical; zero-diagonals; pair-wise distance between samples
    """
    return distance.squareform(distance.pdist(samples, metric=fn))

def Cosine(samples):
    """Calculate a pair-wise distance matrix for a set of samples based on the cosine between their values.

    Args:
        samples (numpy.ndarray): shape (n,p); where n is the number of samples

    Returns:
        numpy.ndarray: shape (n,p); symmetrical; zero-diagonals; pair-wise distance between samples
    """
    return Generic(samples, 'cosine')

def JensenShannon(samples):
    """Calculate a pair-wise distance matrix for a set of samples based on Jensen-Shannon divergence between their values.

    Args:
        samples (numpy.ndarray): shape (n,p); where n is the number of samples

    Returns:
        numpy.ndarray: shape (n,p); symmetrical; zero-diagonals; pair-wise distance between samples
    """
    distFn = lambda P, Q: information.Divergence(P, Q).JS()
    return Generic(samples, fn=distFn)
