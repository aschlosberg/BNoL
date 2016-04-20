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

    def test_discretization_of_continuous_features(self):
        freqs = data.BerrettaExpression()
        (nSamples, nFeatures) = freqs.shape

        D = information.Discretize()
        mdlpCriterionMet = D.fit_transform(freqs, np.asarray([True, False, True, False]), allFeatures=False)

        self.assertTrue(np.alltrue(D._getSeparation(np.arange(1,6), 3)==[False, False, False, True, True]), "Discretization does not properly threshold feature values. MUST be > and not >=.")
        self.assertEqual(D.includeFeatures.dtype, 'bool', "Feature-inclusion array dtype incorrectly defined for discretization by MDLP")
        self.assertEqual(D.includeFeatures.shape, (nFeatures,), "Feature-inclusion array shape incorrectly defined for discretization by MDLP")
        self.assertEqual(D.bestThresholds.dtype, 'float', "Threshold array dtype incorrectly defined for discretization by MDLP")
        self.assertEqual(D.bestThresholds.shape, (nFeatures,), "Threshold array properties shape incorrectly defined for discretization by MDLP")
        self.assertEqual(D.discretizedFeatures.dtype, 'bool', "Discretized-features array dtype incorrectly defined for discretization by MDLP")
        self.assertEqual(D.discretizedFeatures.shape, freqs.shape, "Discretized-features array shape incorrectly defined for discretization by MDLP")
        self.assertTrue(np.alltrue(mdlpCriterionMet==D.discretizedFeatures[:,D.includeFeatures]), "Features returned by Discretize.fit_transform() do not match those chosen explicitly with Discretize.includeFeatures")

        self.assertTrue(np.alltrue(D.bestThresholds==[3.0, 1.5, 3.5, 1.5, 1.05]), "Optimal thresholds returned by Discretize.fit() are incorrect for Berretta data.")
        discretized = np.asarray([
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1]
        ], dtype='bool')
        self.assertTrue(np.alltrue(D.discretizedFeatures==discretized), "Discretized features are incorrect for Berretta data.")

    def test_discretization_of_zero_variance_feature(self):
        """If a single feature has constant value across all specimens there is no threshold that will improve entropy.
        Previously this would result in None being passed as the optimal entropies for delta calculation.
        """
        samples = np.ones((10,1)) * 42
        classes = np.asarray([1,0]*5, dtype='bool') # actual values don't matter for this bug

        D = information.Discretize()
        discrete = D.fit_transform(samples, classes)

        self.assertTrue(np.alltrue(discrete==True), "Zero-variance feature should be discretized to True; arbitrarily chosen") # arbitrarily decided to use True
        self.assertFalse(D.includeFeatures[0], "Zero-variance feature should not be included in discretized values")
        self.assertEqual(D.bestThresholds[0], 42, "Zero-variance feature should have its constant value as best threshold")
        self.assertEqual(D.gains[0], 0, "Zero-variance feature should have zero gain in entropy")
        self.assertGreater(D.mdlpCriteria[0], D.baseEntropy, "Zero-variance feature should have value larger than base entropy as this is impossible to reach")


if __name__=='__main__':
    unittest.main()
