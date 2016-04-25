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
        onlyInformative = analysis.informativeGenes(allGenes=False)
        allGenes = analysis.informativeGenes(allGenes=True)

        nInformative = len(onlyInformative.index)
        ranking = list(map(lambda g: 'GENE_'+g, ['F','D','G','C','I','H','E','A','B']))

        self.assertTrue(np.alltrue(allGenes.index==ranking), "Incorrect ranking of genes by entropy gain")
        self.assertTrue(np.alltrue(onlyInformative.index==allGenes.index[:nInformative]), "Incorrect subset of genes in informative grouping")
        self.assertTrue(np.alltrue(onlyInformative==allGenes.iloc[:nInformative,:]), "Values for informative genes do not match across informative / all-genes groups")
        self.assertTrue(np.alltrue(onlyInformative.loc[:,'Informative']==1), "Non-informative genes included in informative-only group")
        self.assertTrue(np.alltrue(allGenes.loc[:,'Informative'][nInformative:]==0), "Informative genes included in non-informative section")
        informative = allGenes.loc[:,'Gain']>allGenes.loc[:,'MDLP-Criterion']
        self.assertTrue(np.alltrue(informative==allGenes.loc[:,'Informative']), "Gain vs MDLP-criterion does not match marking as informative")

if __name__=='__main__':
    unittest.main()
