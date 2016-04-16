import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from bnol import utility, information, data
import unittest
import numpy as np

class InformationTester(unittest.TestCase):

    def test_entropy_calculation(self):
        freqs = data.BerrettaExpression()
        berrettaEntropy = information.Entropy(freqs)

        self.assertEqual(berrettaEntropy.shape, (freqs.shape[0],), "Specimen entropies calculated across incorrect axis")
        self.assertTrue(np.isclose(berrettaEntropy[0], berrettaEntropy[1], data.epsilon()), "Permutations of the same distribution should have the same entropy")
        self.assertEqual(np.argmax(berrettaEntropy), freqs.shape[0]-1, "Uniform distribution should have the maximum possible entropy")

        for n in range(1,25):
            uniform = utility.DiscreteUniform(n)
            self.assertTrue(np.isclose(information.Entropy(uniform), np.log2(n), data.epsilon()), "Incorrect entropy of discrete uniform distribution for n=%d" % n)

    def test_divergence_from_uniform_distribution(self):
        freqs = data.BerrettaExpression()
        uniformDivergence = information.Divergence(Qs=freqs)

        for measure in ['KL', 'JS']:
            vals = getattr(uniformDivergence, measure)()
            self.assertEqual(vals[-1], 0, "Divergence (%s) from self (tested as uniform distribution) should be zero" % measure)
            self.assertEqual(vals[0], vals[1], "Divergence (%s) of permutations from uniform distribution should be the same" % measure)

    def test_statistical_complexity_compared_to_uniform_distribution(self):
        freqs = data.BerrettaExpression()

        complexity = information.Complexity(freqs, utility.DiscreteUniform(freqs.shape[1]))
        self.assertEqual(len(complexity), freqs.shape[0], "Complexity calculated across incorrect axis")
        self.assertTrue(np.isclose(complexity[0], complexity[1], data.epsilon()), "Complexity of permutations should be the same when compared to uniform reference")
        self.assertEqual(complexity[-1], 0, "Complexity should be zero when compared to self")

if __name__=='__main__':
    unittest.main()
