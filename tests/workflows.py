import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from bnol import workflows
import unittest
import numpy as np

class WorflowsTester(unittest.TestCase):

    @staticmethod
    def _testFilePath(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', filename)

    def test_cuffnorm_one_vs_rest_analysis(self):
        analysis = workflows.CuffnormOneVsRest(self._testFilePath('genes.count_table'), [True]*30 + [False]*89)
        informative = analysis.informativeGenes()

if __name__=='__main__':
    unittest.main()
