import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from bnol import utility, information, data
import unittest
import numpy as np
import base64

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

        self.assertTrue(np.alltrue(D.getSeparation(np.arange(1,6), 3)==[False, False, False, True, True]), "Discretization does not properly threshold feature values. MUST be > and not >=.")
        self.assertEqual(D.includeFeatures.dtype, 'bool', "Feature-inclusion array dtype incorrectly defined for discretization by MDLP")
        self.assertEqual(D.includeFeatures.shape, (nFeatures,), "Feature-inclusion array shape incorrectly defined for discretization by MDLP")
        self.assertTrue(np.alltrue(D.includeFeatures==(D.gains>D.mdlpCriteria)), "Feature-inclusion array does not match that determined by comparing gains array to MDLP criteria array")
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

    def test_mdlp_criterion_for_discretization(self):
        """Series of test tuples::
            (
                boolean-flag list indicating if each of the features is above threshold,
                2D list of specimen classes for each of the two separations defined by the above flags,
                list of expected entropy values of each of above list,
                expected delta value for MDLP criterion
            )
        """

        D = information.Discretize()
        classes = np.asarray([1,0,0,1,1], dtype='bool')
        D.fit(np.random.random_sample((5,6)), classes)

        ents = np.zeros((5,5))
        for i in range(1,5):
            for j in range(1,5):
                ents[i,j] = information.Entropy([i,j])

        c = np.log2(6) - 2*ents[2,3] # constant value for all deltas
        tests = [
            ([0,1,1,1,1], [[0,0,1,1],[1]], [ents[2,2], ents[1,0]], c + 2*ents[2,2] + 1*ents[1,0]),
            ([1,0,1,1,1], [[1,0,1,1],[0]], [ents[3,1], ents[0,1]], c + 2*ents[3,1] + 1*ents[0,1]),
            ([1,1,0,1,1], [[1,0,1,1],[0]], [ents[3,1], ents[0,1]], c + 2*ents[3,1] + 1*ents[0,1]),
            ([1,1,1,0,1], [[1,0,0,1],[1]], [ents[2,2], ents[1,0]], c + 2*ents[2,2] + 1*ents[1,0]),
            ([1,1,1,1,0], [[1,0,0,1],[1]], [ents[2,2], ents[1,0]], c + 2*ents[2,2] + 1*ents[1,0]),
            ([0,0,1,1,1], [[0,1,1],[1,0]], [ents[2,1], ents[1,1]], c + 2*ents[1,2] + 2*ents[1,1]),
            ([0,1,0,1,1], [[0,1,1],[1,0]], [ents[2,1], ents[1,1]], c + 2*ents[2,1] + 2*ents[1,1]),
            ([0,1,1,0,1], [[0,0,1],[1,1]], [ents[1,2], ents[2,0]], c + 2*ents[1,2] + 1*ents[2,0]),
            ([0,1,1,1,0], [[0,0,1],[1,1]], [ents[1,2], ents[2,0]], c + 2*ents[1,2] + 2*ents[2,0]),
            ([1,0,0,1,1], [[1,1,1],[0,0]], [ents[3,0], ents[0,2]], c + 1*ents[3,0] + 1*ents[0,2]),
            ([1,0,1,0,1], [[1,0,1],[0,1]], [ents[2,1], ents[1,1]], c + 2*ents[2,1] + 2*ents[1,1]),
            ([1,0,1,1,0], [[1,0,1],[0,1]], [ents[2,1], ents[1,1]], c + 2*ents[2,1] + 2*ents[1,1]),
            ([1,1,0,0,1], [[1,0,1],[0,1]], [ents[2,1], ents[1,1]], c + 2*ents[2,1] + 2*ents[1,1]),
            ([1,1,0,1,0], [[1,0,1],[0,1]], [ents[2,1], ents[1,1]], c + 2*ents[2,1] + 2*ents[1,1]),
            ([1,1,1,0,0], [[1,0,0],[1,1]], [ents[1,2], ents[2,0]], c + 2*ents[1,2] + 1*ents[2,0]),
        ]

        for i, t in enumerate(tests):
            above = np.asarray(t[0], dtype='bool')
            separations = [classes[a] for a in [above, ~above]]
            entropies = [D.specimenClassEntropy(s) for s in separations]

            for j in range(2):
                self.assertTrue(np.alltrue(separations[j]==t[1][j]), "Separation of boolean classes incorrectly calculated for testing of MDLP criterion; test [%d] %s" % (i, t[0]))
                self.assertTrue(np.isclose(entropies[j], t[2][j], data.epsilon()), "Entropies of separation of boolean classes incorrectly calculated for testing of MDLP criterion; test [%d] %s" % (i, t[0]))
            self.assertTrue(np.isclose(t[3], D.deltaMDLP(classes, ents[3,2], above, entropies), data.epsilon()), "Incorrect delta value for MDLP criterion; test [%d] %s" % (i, t[0]))

    def test_discretization_of_zero_variance_feature(self):
        """If a single feature has constant value across all specimens there is no threshold that will improve entropy.
        Previously this would result in None being passed as the optimal entropies for delta calculation.
        """
        samples = np.ones((10,1)) * 42
        classes = np.asarray([1,0]*5, dtype='bool') # actual values don't matter for this bug

        D = information.Discretize()
        D.fit(samples, classes)
        discrete = D.discretizedFeatures

        self.assertEqual(discrete.shape, samples.shape, "Zero-variance feature discretization can not be checked if using Discretize.fit_transform() - must use discretizedFeatures attribute")
        self.assertTrue(np.alltrue(discrete==True), "Zero-variance feature should be discretized to True; arbitrarily chosen") # arbitrarily decided to use True
        self.assertFalse(D.includeFeatures[0], "Zero-variance feature should not be included in discretized values")
        self.assertEqual(D.bestThresholds[0], 42, "Zero-variance feature should have its constant value as best threshold")
        self.assertEqual(D.gains[0], 0, "Zero-variance feature should have zero gain in entropy")
        self.assertGreater(D.mdlpCriteria[0], D.baseEntropy, "Zero-variance feature should have value larger than base entropy as this is impossible to reach")

    def test_parallel_vs_single_process_discretization(self):
        """Assuming that the single-process discretization implementation is correct at this point, save some outcomes to compare against the parallelised version.
        Randomly generate a series of specimens, seeded with nothing-up-my-sleeve type numbers, and then compute their discretization.
        """
        seeds = [0, 42, 314159, 0xD15EA5E]
        version = sys.version_info[0]
        if version==2:
            expected = [{'discretizedFeatures': 'AQEBAQEBAAEAAQABAAEBAQEBAQAAAAEBAQABAAEBAQEBAQEBAAEAAQABAAEAAQEBAQABAQEAAQAAAAEBAQAAAQAAAQABAAABAAAAAQABAQEBAAEBAAABAAEBAAABAAABAQEBAAAAAAEAAQABAQABAAEAAQABAAEBAQEBAAEBAAEAAAABAAEBAQEBAQEBAQEAAAABAAEBAQABAAAAAQEBAAEAAQEAAQEBAQAAAQABAAABAAEA', 'mdlpCriteria': 'jE9M4Snh3j9JKAsCUPjgP4xPTOEp4d4/+E5dX4lf4z/sodZezgvhP+cA6Q0fBuQ/VwPagvkf4T8aPU2rpfnkP1cD2oL5H+E/7KHWXs4L4T+MT0zhKeHeP/hOXV+JX+M/jE9M4Snh3j+MT0zhKeHeP0koCwJQ+OA/', 'bestThresholds': 'YgAAAAAAAAC7AAAAAAAAAJ4AAAAAAAAAugEAAAAAAACDAgAAAAAAAMAAAAAAAAAA/wEAAAAAAAB9AAAAAAAAAJ4CAAAAAAAAZQIAAAAAAACaAwAAAAAAAFQCAAAAAAAAnQMAAAAAAABiAQAAAAAAAJkBAAAAAAAA', 'gains': 'rBhhZUWVxz9eZIDunwfaP6wYYWVFlcc/KInajDQlyj+4OMZ3p0fOP7AXTp3llb0/xscgFfvr0z/w266vtgSmP8bHIBX769M/uDjGd6dHzj+sGGFlRZXHPyiJ2ow0Jco/rBhhZUWVxz+sGGFlRZXHP15kgO6fB9o/', 'baseEntropy': '42OQiv316T8=', 'includeFeatures': 'AAAAAAAAAAAAAAAAAAAA'}, {'discretizedFeatures': 'AQAAAAEAAQEBAAABAQAAAQAAAQABAAABAQEBAQAAAQEAAAEBAAAAAQEAAQEBAQEBAQEBAQABAAABAQEAAQEBAQEBAAABAQEBAQAAAQEAAQEBAAAAAQEBAAABAQABAQEBAAEBAAABAQABAQABAQEAAQAB', 'mdlpCriteria': 'RUo47KUK6z9tATGSiivqP20BMZKKK+o/bQExkoor6j8gE3SSUxPoP20BMZKKK+o/IBN0klMT6D8gE3SSUxPoPyATdJJTE+g/bQExkoor6j9tATGSiivqP5ufjUH7X+A/IBN0klMT6D9FSjjspQrrP20BMZKKK+o/m5+NQftf4D9tATGSiivqPyATdJJTE+g/K5Sa6lK76j8=', 'bestThresholds': 'fQEAAAAAAADUAQAAAAAAAKkBAAAAAAAACwIAAAAAAACjAAAAAAAAAO0AAAAAAAAAJwEAAAAAAABkAAAAAAAAACQBAAAAAAAAogEAAAAAAAA8AgAAAAAAAKYBAAAAAAAALAAAAAAAAAB8AQAAAAAAAJgBAAAAAAAAJwEAAAAAAACqAgAAAAAAALICAAAAAAAA2wEAAAAAAAA=', 'gains': 'TKApObEa0D97JerxrWLdP3sl6vGtYt0/eyXq8a1i3T9YCVRPokTUP3sl6vGtYt0/WAlUT6JE1D9YCVRPokTUP1gJVE+iRNQ/eyXq8a1i3T97JerxrWLdP3sl6vGtYu0/WAlUT6JE1D9MoCk5sRrQP3sl6vGtYt0/eyXq8a1i7T97JerxrWLdP1gJVE+iRNQ/mBBOWpbyuz8=', 'baseEntropy': 'eyXq8a1i7T8=', 'includeFeatures': 'AAAAAAAAAAAAAAABAAAAAQAAAA=='}, {'discretizedFeatures': 'AQEBAAEBAQEBAQEBAQEBAQABAQABAQ==', 'mdlpCriteria': 'QFrqf6UM4T/z+tBYnyrhPw==', 'bestThresholds': 'XQAAAAAAAABWAQAAAAAAAA==', 'gains': 'GG+b+kqBvD/Q0jvec3HHPw==', 'baseEntropy': 'mvEwchjP7z8=', 'includeFeatures': 'AAA='}, {'discretizedFeatures': 'AQEBAAEBAQEBAQEBAAEBAAEBAQEBAAEBAAEBAQEBAAEBAQEBAQABAQEBAQABAAEBAAEBAQEAAQEBAAEBAQEBAQEBAAEBAQEBAQEBAAEA', 'mdlpCriteria': '4ihw41sa1T+CSzutcafRPy8c7vGM2tE/', 'bestThresholds': 'TAIAAAAAAACdAAAAAAAAAHAAAAAAAAAA', 'gains': 'LP/ZL8cPwT8QdQ7CXBK6P4hUh+A/fbA/', 'baseEntropy': 't/kcEZRz7z8=', 'includeFeatures': 'AAAA'}]
        elif version==3: # same data but different base64.b64encode output format
            expected = [{'mdlpCriteria': b'jE9M4Snh3j9JKAsCUPjgP4xPTOEp4d4/+E5dX4lf4z/sodZezgvhP+cA6Q0fBuQ/VwPagvkf4T8aPU2rpfnkP1cD2oL5H+E/7KHWXs4L4T+MT0zhKeHeP/hOXV+JX+M/jE9M4Snh3j+MT0zhKeHeP0koCwJQ+OA/', 'gains': b'rBhhZUWVxz9eZIDunwfaP6wYYWVFlcc/KInajDQlyj+4OMZ3p0fOP7AXTp3llb0/xscgFfvr0z/w266vtgSmP8bHIBX769M/uDjGd6dHzj+sGGFlRZXHPyiJ2ow0Jco/rBhhZUWVxz+sGGFlRZXHP15kgO6fB9o/', 'discretizedFeatures': b'AQEBAQEBAAEAAQABAAEBAQEBAQAAAAEBAQABAAEBAQEBAQEBAAEAAQABAAEAAQEBAQABAQEAAQAAAAEBAQAAAQAAAQABAAABAAAAAQABAQEBAAEBAAABAAEBAAABAAABAQEBAAAAAAEAAQABAQABAAEAAQABAAEBAQEBAAEBAAEAAAABAAEBAQEBAQEBAQEAAAABAAEBAQABAAAAAQEBAAEAAQEAAQEBAQAAAQABAAABAAEA', 'baseEntropy': b'42OQiv316T8=', 'bestThresholds': b'AAAAAACgWEAAAAAAAHBnQAAAAAAAwGNAAAAAAACge0AAAAAAABiEQAAAAAAAEGhAAAAAAADwf0AAAAAAAEBfQAAAAAAA8IRAAAAAAAAsg0AAAAAAANCMQAAAAAAApIJAAAAAAADsjEAAAAAAACB2QAAAAAAAkHlA', 'includeFeatures': b'AAAAAAAAAAAAAAAAAAAA'}, {'mdlpCriteria': b'RUo47KUK6z9tATGSiivqP20BMZKKK+o/bQExkoor6j8gE3SSUxPoP20BMZKKK+o/IBN0klMT6D8gE3SSUxPoPyATdJJTE+g/bQExkoor6j9tATGSiivqP5ufjUH7X+A/IBN0klMT6D9FSjjspQrrP20BMZKKK+o/m5+NQftf4D9tATGSiivqPyATdJJTE+g/K5Sa6lK76j8=', 'gains': b'TKApObEa0D97JerxrWLdP3sl6vGtYt0/eyXq8a1i3T9YCVRPokTUP3sl6vGtYt0/WAlUT6JE1D9YCVRPokTUP1gJVE+iRNQ/eyXq8a1i3T97JerxrWLdP3sl6vGtYu0/WAlUT6JE1D9MoCk5sRrQP3sl6vGtYt0/eyXq8a1i7T97JerxrWLdP1gJVE+iRNQ/mBBOWpbyuz8=', 'discretizedFeatures': b'AQAAAAEAAQEBAAABAQAAAQAAAQABAAABAQEBAQAAAQEAAAEBAAAAAQEAAQEBAQEBAQEBAQABAAABAQEAAQEBAQEBAAABAQEBAQAAAQEAAQEBAAAAAQEBAAABAQABAQEBAAEBAAABAQABAQABAQEAAQAB', 'baseEntropy': b'eyXq8a1i7T8=', 'bestThresholds': b'AAAAAADQd0AAAAAAAEh9QAAAAAAAmHpAAAAAAABYgEAAAAAAAHBkQAAAAAAAoG1AAAAAAABwckAAAAAAACBZQAAAAAAAQHJAAAAAAAAoekAAAAAAAOSBQAAAAAAAaHpAAAAAAAAARkAAAAAAAMh3QAAAAAAAiHlAAAAAAABwckAAAAAAAFCFQAAAAAAAlIVAAAAAAACwfUA=', 'includeFeatures': b'AAAAAAAAAAAAAAABAAAAAQAAAA=='}, {'mdlpCriteria': b'QFrqf6UM4T/z+tBYnyrhPw==', 'gains': b'GG+b+kqBvD/Q0jvec3HHPw==', 'discretizedFeatures': b'AQEBAAEBAQEBAQEBAQEBAQABAQABAQ==', 'baseEntropy': b'mvEwchjP7z8=', 'bestThresholds': b'AAAAAABgV0AAAAAAAGB1QA==', 'includeFeatures': b'AAA='}, {'mdlpCriteria': b'4ihw41sa1T+CSzutcafRPy8c7vGM2tE/', 'gains': b'LP/ZL8cPwT8QdQ7CXBK6P4hUh+A/fbA/', 'discretizedFeatures': b'AQEBAAEBAQEBAQEBAAEBAAEBAQEBAAEBAAEBAQEBAAEBAQEBAQABAQEBAQABAAEBAAEBAQEAAQEBAAEBAQEBAQEBAAEBAQEBAQEBAAEA', 'baseEntropy': b't/kcEZRz7z8=', 'bestThresholds': b'AAAAAABkgkAAAAAAALBjQAAAAAAAAFxA', 'includeFeatures': b'AAAA'}]

        # Originally classes were assigned randomly with np.random.randint and dtype='bool' but this caused errors in python3.3; have hardcoded them isntead
        # See https://travis-ci.org/aschlosberg/BNoL/jobs/124722492
        allClasses = [[0,0,0,0,1,0,1,0,0,0,1,0],[1,1,1,1,0,0],[1,0,1,0,1,0,0,0,1,0,1],[1,0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,1,1,0]]

        for i, seed in enumerate(seeds):
            np.random.seed(seed)
            nSamples = np.random.randint(30)
            nFeatures = np.random.randint(20)
            specimens = np.random.randint(1e3, size=(nSamples, nFeatures))
            classes = np.asarray(allClasses[i], dtype='bool')
            D = information.Discretize()
            D.fit(specimens, classes)

            data = {
                'baseEntropy' : D.baseEntropy,
                'discretizedFeatures' : D.discretizedFeatures,
                'includeFeatures' : D.includeFeatures,
                'bestThresholds' : D.bestThresholds,
                'gains' : D.gains,
                'mdlpCriteria' : D.mdlpCriteria,
            }
            for d in data:
                b64 = base64.b64encode(data[d].tostring(order='C') if type(data[d])==type(np.ndarray) else data[d].tostring())
                self.assertEqual(b64, expected[i][d], "Incorrect value for information.Discretize attribute '%s' when using random seed %d" % (d, seeds[i]))

if __name__=='__main__':
    unittest.main()
