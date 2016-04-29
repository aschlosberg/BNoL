import numpy as np
from scipy.spatial import distance
from math import sqrt

def Cosine(samples):
    """Calculate a pair-wise distance matrix for a set of samples based on the cosine between their values.

    Args:
        samples (numpy.ndarray): shape (n,p); where n is the number of samples

    Returns:
        numpy.ndarray: shape (n,p); symmetrical; zero-diagonals; pair-wise distance between samples
    """
    nSamples = samples.shape[0]
    similarity = np.zeros((nSamples, nSamples), dtype='float')
    for i in range(nSamples):
        for j in range(i):
            similarity[i,j] = distance.cosine(samples[i,:], samples[j,:])
            similarity[j,i] = similarity[i,j]
    return similarity
