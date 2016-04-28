import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from bnol import utility, information, data
import unittest
from numpyfied import NumpyfiedTestCase
import numpy as np
import base64

class InformationTester(NumpyfiedTestCase):

    def test_entropy_calculation(self):
        freqs = data.BerrettaExpression()
        berrettaEntropy = information.Entropy(freqs)

        self.assertEqual(berrettaEntropy.shape, (freqs.shape[0],), "Specimen entropies calculated across incorrect axis")
        self.assertIsClose(berrettaEntropy[0], berrettaEntropy[1], "Permutations of the same distribution should have the same entropy")
        self.assertEqual(np.argmax(berrettaEntropy), freqs.shape[0]-1, "Uniform distribution should have the maximum possible entropy")

        for n in range(1,25):
            uniform = utility.DiscreteUniform(n)
            self.assertIsClose(information.Entropy(uniform), np.log2(n), "Incorrect entropy of discrete uniform distribution for n=%d" % n)

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
        self.assertIsClose(complexity[0], complexity[1], "Complexity of permutations should be the same when compared to uniform reference")
        self.assertEqual(complexity[-1], 0, "Complexity should be zero when compared to self")

    def test_discretization_of_continuous_features(self):
        freqs = data.BerrettaExpression()
        (nSamples, nFeatures) = freqs.shape

        D = information.Discretize()
        mdlpCriterionMet = D.fit_transform(freqs, np.asarray([True, False, True, False]), allFeatures=False)

        self.assertAllTrue(D.getSeparation(np.arange(1,6), 3)==[False, False, False, True, True], "Discretization does not properly threshold feature values. MUST be > and not >=.")
        self.assertEqual(D.includeFeatures.dtype, 'bool', "Feature-inclusion array dtype incorrectly defined for discretization by MDLP")
        self.assertEqual(D.includeFeatures.shape, (nFeatures,), "Feature-inclusion array shape incorrectly defined for discretization by MDLP")
        self.assertAllTrue(D.includeFeatures==(D.gains>D.mdlpCriteria), "Feature-inclusion array does not match that determined by comparing gains array to MDLP criteria array")
        self.assertEqual(D.bestThresholds.dtype, 'float', "Threshold array dtype incorrectly defined for discretization by MDLP")
        self.assertEqual(D.bestThresholds.shape, (nFeatures,), "Threshold array properties shape incorrectly defined for discretization by MDLP")
        self.assertEqual(D.discretizedFeatures.dtype, 'bool', "Discretized-features array dtype incorrectly defined for discretization by MDLP")
        self.assertEqual(D.discretizedFeatures.shape, freqs.shape, "Discretized-features array shape incorrectly defined for discretization by MDLP")
        self.assertAllTrue(mdlpCriterionMet==D.discretizedFeatures[:,D.includeFeatures], "Features returned by Discretize.fit_transform() do not match those chosen explicitly with Discretize.includeFeatures")

        self.assertAllTrue(D.bestThresholds==[3.0, 1.5, 3.5, 1.5, 1.05], "Optimal thresholds returned by Discretize.fit() are incorrect for Berretta data.")
        discretized = np.asarray([
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1]
        ], dtype='bool')
        self.assertAllTrue(D.discretizedFeatures==discretized, "Discretized features are incorrect for Berretta data.")

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

        c = np.log2(7) - 2*ents[2,3] # constant value for all deltas
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
            entropies = [D.groupClassEntropy(s) for s in separations]

            for j in range(2):
                self.assertAllTrue(separations[j]==t[1][j], "Separation of boolean classes incorrectly calculated for testing of MDLP criterion; test [%d] %s" % (i, t[0]))
                self.assertIsClose(entropies[j], t[2][j], "Entropies of separation of boolean classes incorrectly calculated for testing of MDLP criterion; test [%d] %s" % (i, t[0]))
            self.assertIsClose(t[3], D.deltaMDLP(classes, ents[3,2], above, entropies), "Incorrect delta value for MDLP criterion; test [%d] %s" % (i, t[0]))

    def test_correct_combinatorial_term_in_delta_mdlp(self):
        """Fayyad and Irani define a delta term for calculation of MDLP that includes a count of combinations (3^k - 2) where k=2 for this work. Was incorrectly implemented as (2^3 - 2)."""

        # dummy classes and separation simply so that the function runs as expected
        classes = np.asarray([1,0], dtype='bool')
        separation = np.asarray([1,0], dtype='bool')

        # explicitly set all entropy values to zero (even thought they are not the true values) so that we can get only the combinatorial term (3^k - 2)
        baseEntropy = 0
        separationEntropies = [0,0]

        combinatorialTerm = information.Discretize.deltaMDLP(classes, 0, separation, [0,0])
        self.assertEqual(combinatorialTerm, np.log2(3**2 - 2), "Discretization delta has been calculated with incorrect count of combinations")

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
        self.assertAllTrue(discrete==True, "Zero-variance feature should be discretized to True; arbitrarily chosen") # arbitrarily decided to use True
        self.assertFalse(D.includeFeatures[0], "Zero-variance feature should not be included in discretized values")
        self.assertEqual(D.bestThresholds[0], 42, "Zero-variance feature should have its constant value as best threshold")
        self.assertEqual(D.gains[0], 0, "Zero-variance feature should have zero gain in entropy")
        self.assertGreater(D.mdlpCriteria[0], D.baseEntropy, "Zero-variance feature should have value larger than base entropy as this is impossible to reach")

    def test_parallel_vs_single_process_discretization(self):
        """Assuming that the single-process discretization implementation is correct at this point, save some outcomes to compare against the parallelised version.
        Randomly generate a series of specimens, seeded with nothing-up-my-sleeve type numbers, and then compute their discretization.
        The values were corrected for mdlpCriteria only when the bug addressed by test_correct_combinatorial_term_in_delta_mdlp() was fixed. It was expected that these would be incorrect and all others unchanged.
        """
        seeds = [0, 42, 314159, 0xD15EA5E]
        version = sys.version_info[0]
        if version==2:
            expected = [{'discretizedFeatures': 'AQEBAQEBAAEAAQABAAEBAQEBAQAAAAEBAQABAAEBAQEBAQEBAAEAAQABAAEAAQEBAQABAQEAAQAAAAEBAQAAAQAAAQABAAABAAAAAQABAQEBAAEBAAABAAEBAAABAAABAQEBAAAAAAEAAQABAQABAAEAAQABAAEBAQEBAAEBAAEAAAABAAEBAQEBAQEBAQEAAAABAAEBAQABAAAAAQEBAAEAAQEAAQEBAQAAAQABAAABAAEA', 'mdlpCriteria': 'mYkn1WYI4D8diozmIZDhP5mJJ9VmCOA/y7DeQ1v34z+/A1hDoKPhP7tiavLwneQ/KmVbZ8u34T/tns6Pd5HlPyplW2fLt+E/vwNYQ6Cj4T+ZiSfVZgjgP8uw3kNb9+M/mYkn1WYI4D+ZiSfVZgjgPx2KjOYhkOE/', 'bestThresholds': 'YgAAAAAAAAC7AAAAAAAAAJ4AAAAAAAAAugEAAAAAAACDAgAAAAAAAMAAAAAAAAAA/wEAAAAAAAB9AAAAAAAAAJ4CAAAAAAAAZQIAAAAAAACaAwAAAAAAAFQCAAAAAAAAnQMAAAAAAABiAQAAAAAAAJkBAAAAAAAA', 'gains': 'rBhhZUWVxz9eZIDunwfaP6wYYWVFlcc/KInajDQlyj+4OMZ3p0fOP7AXTp3llb0/xscgFfvr0z/w266vtgSmP8bHIBX769M/uDjGd6dHzj+sGGFlRZXHPyiJ2ow0Jco/rBhhZUWVxz+sGGFlRZXHP15kgO6fB9o/', 'baseEntropy': '42OQiv316T8=', 'includeFeatures': 'AAAAAAAAAAAAAAAAAAAA'}, {'discretizedFeatures': 'AQAAAAEAAQEBAAABAQAAAQAAAQABAAABAQEBAQAAAQEAAAEBAAAAAQEAAQEBAQEBAQEBAQABAAABAQEAAQEBAQEBAAABAQEBAQAAAQEAAQEBAAAAAQEBAAABAQABAQEBAAEBAAABAQABAQABAQEAAQAB', 'mdlpCriteria': '6w07tUk67D8UxTNbLlvrPxTFM1suW+s/FMUzWy5b6z/F1nZb90LpPxTFM1suW+s/xdZ2W/dC6T/F1nZb90LpP8XWdlv3Quk/FMUzWy5b6z8UxTNbLlvrP0BjkAqfj+E/xdZ2W/dC6T/rDTu1STrsPxTFM1suW+s/QGOQCp+P4T8UxTNbLlvrP8XWdlv3Quk/0Feds/bq6z8=', 'bestThresholds': 'fQEAAAAAAADUAQAAAAAAAKkBAAAAAAAACwIAAAAAAACjAAAAAAAAAO0AAAAAAAAAJwEAAAAAAABkAAAAAAAAACQBAAAAAAAAogEAAAAAAAA8AgAAAAAAAKYBAAAAAAAALAAAAAAAAAB8AQAAAAAAAJgBAAAAAAAAJwEAAAAAAACqAgAAAAAAALICAAAAAAAA2wEAAAAAAAA=', 'gains': 'TKApObEa0D97JerxrWLdP3sl6vGtYt0/eyXq8a1i3T9YCVRPokTUP3sl6vGtYt0/WAlUT6JE1D9YCVRPokTUP1gJVE+iRNQ/eyXq8a1i3T97JerxrWLdP3sl6vGtYu0/WAlUT6JE1D9MoCk5sRrQP3sl6vGtYt0/eyXq8a1i7T97JerxrWLdP1gJVE+iRNQ/mBBOWpbyuz8=', 'baseEntropy': 'eyXq8a1i7T8=', 'includeFeatures': 'AAAAAAAAAAAAAAABAAAAAQAAAA=='}, {'discretizedFeatures': 'AQEBAAEBAQEBAQEBAQEBAQABAQABAQ==', 'mdlpCriteria': 'mlC9p0Sy4T9O8aOAPtDhPw==', 'bestThresholds': 'XQAAAAAAAABWAQAAAAAAAA==', 'gains': 'GG+b+kqBvD/Q0jvec3HHPw==', 'baseEntropy': 'mvEwchjP7z8=', 'includeFeatures': 'AAA='}, {'discretizedFeatures': 'AQEBAAEBAQEBAQEBAAEBAAEBAQEBAAEBAAEBAQEBAAEBAQEBAQABAQEBAQABAAEBAAEBAQEAAQEBAAEBAQEBAQEBAAEBAQEBAQEBAAEA', 'mdlpCriteria': 'peXTGICm1T9FCJ/ilTPSP/LYUSexZtI/', 'bestThresholds': 'TAIAAAAAAACdAAAAAAAAAHAAAAAAAAAA', 'gains': 'LP/ZL8cPwT8QdQ7CXBK6P4hUh+A/fbA/', 'baseEntropy': 't/kcEZRz7z8=', 'includeFeatures': 'AAAA'}]
        elif version==3: # same data but different base64.b64encode output format
            expected = [{'discretizedFeatures': b'AQEBAQEBAAEAAQABAAEBAQEBAQAAAAEBAQABAAEBAQEBAQEBAAEAAQABAAEAAQEBAQABAQEAAQAAAAEBAQAAAQAAAQABAAABAAAAAQABAQEBAAEBAAABAAEBAAABAAABAQEBAAAAAAEAAQABAQABAAEAAQABAAEBAQEBAAEBAAEAAAABAAEBAQEBAQEBAQEAAAABAAEBAQABAAAAAQEBAAEAAQEAAQEBAQAAAQABAAABAAEA', 'mdlpCriteria': b'mYkn1WYI4D8diozmIZDhP5mJJ9VmCOA/y7DeQ1v34z+/A1hDoKPhP7tiavLwneQ/KmVbZ8u34T/tns6Pd5HlPyplW2fLt+E/vwNYQ6Cj4T+ZiSfVZgjgP8uw3kNb9+M/mYkn1WYI4D+ZiSfVZgjgPx2KjOYhkOE/', 'gains': b'rBhhZUWVxz9eZIDunwfaP6wYYWVFlcc/KInajDQlyj+4OMZ3p0fOP7AXTp3llb0/xscgFfvr0z/w266vtgSmP8bHIBX769M/uDjGd6dHzj+sGGFlRZXHPyiJ2ow0Jco/rBhhZUWVxz+sGGFlRZXHP15kgO6fB9o/', 'baseEntropy': b'42OQiv316T8=', 'includeFeatures': b'AAAAAAAAAAAAAAAAAAAA', 'bestThresholds': b'AAAAAACgWEAAAAAAAHBnQAAAAAAAwGNAAAAAAACge0AAAAAAABiEQAAAAAAAEGhAAAAAAADwf0AAAAAAAEBfQAAAAAAA8IRAAAAAAAAsg0AAAAAAANCMQAAAAAAApIJAAAAAAADsjEAAAAAAACB2QAAAAAAAkHlA'}, {'discretizedFeatures': b'AQAAAAEAAQEBAAABAQAAAQAAAQABAAABAQEBAQAAAQEAAAEBAAAAAQEAAQEBAQEBAQEBAQABAAABAQEAAQEBAQEBAAABAQEBAQAAAQEAAQEBAAAAAQEBAAABAQABAQEBAAEBAAABAQABAQABAQEAAQAB', 'mdlpCriteria': b'6w07tUk67D8UxTNbLlvrPxTFM1suW+s/FMUzWy5b6z/F1nZb90LpPxTFM1suW+s/xdZ2W/dC6T/F1nZb90LpP8XWdlv3Quk/FMUzWy5b6z8UxTNbLlvrP0BjkAqfj+E/xdZ2W/dC6T/rDTu1STrsPxTFM1suW+s/QGOQCp+P4T8UxTNbLlvrP8XWdlv3Quk/0Feds/bq6z8=', 'gains': b'TKApObEa0D97JerxrWLdP3sl6vGtYt0/eyXq8a1i3T9YCVRPokTUP3sl6vGtYt0/WAlUT6JE1D9YCVRPokTUP1gJVE+iRNQ/eyXq8a1i3T97JerxrWLdP3sl6vGtYu0/WAlUT6JE1D9MoCk5sRrQP3sl6vGtYt0/eyXq8a1i7T97JerxrWLdP1gJVE+iRNQ/mBBOWpbyuz8=', 'baseEntropy': b'eyXq8a1i7T8=', 'includeFeatures': b'AAAAAAAAAAAAAAABAAAAAQAAAA==', 'bestThresholds': b'AAAAAADQd0AAAAAAAEh9QAAAAAAAmHpAAAAAAABYgEAAAAAAAHBkQAAAAAAAoG1AAAAAAABwckAAAAAAACBZQAAAAAAAQHJAAAAAAAAoekAAAAAAAOSBQAAAAAAAaHpAAAAAAAAARkAAAAAAAMh3QAAAAAAAiHlAAAAAAABwckAAAAAAAFCFQAAAAAAAlIVAAAAAAACwfUA='}, {'discretizedFeatures': b'AQEBAAEBAQEBAQEBAQEBAQABAQABAQ==', 'mdlpCriteria': b'mlC9p0Sy4T9O8aOAPtDhPw==', 'gains': b'GG+b+kqBvD/Q0jvec3HHPw==', 'baseEntropy': b'mvEwchjP7z8=', 'includeFeatures': b'AAA=', 'bestThresholds': b'AAAAAABgV0AAAAAAAGB1QA=='}, {'discretizedFeatures': b'AQEBAAEBAQEBAQEBAAEBAAEBAQEBAAEBAAEBAQEBAAEBAQEBAQABAQEBAQABAAEBAAEBAQEAAQEBAAEBAQEBAQEBAAEBAQEBAQEBAAEA', 'mdlpCriteria': b'peXTGICm1T9FCJ/ilTPSP/LYUSexZtI/', 'gains': b'LP/ZL8cPwT8QdQ7CXBK6P4hUh+A/fbA/', 'baseEntropy': b't/kcEZRz7z8=', 'includeFeatures': b'AAAA', 'bestThresholds': b'AAAAAABkgkAAAAAAALBjQAAAAAAAAFxA'}]

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
