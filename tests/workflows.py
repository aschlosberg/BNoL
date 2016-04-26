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
        ranking = ["GENE_%s" % g for g in ['F','D','G','C','I','H','E','A','B']]

        self.assertTrue(np.alltrue(allGenes.index==ranking), "Incorrect ranking of genes by entropy gain")
        self.assertTrue(np.alltrue(onlyInformative.index==allGenes.index[:nInformative]), "Incorrect subset of genes in informative grouping")
        self.assertTrue(np.alltrue(onlyInformative==allGenes.iloc[:nInformative,:]), "Values for informative genes do not match across informative / all-genes groups")
        self.assertTrue(np.alltrue(onlyInformative.loc[:,'Informative']==1), "Non-informative genes included in informative-only group")
        self.assertTrue(np.alltrue(allGenes.loc[:,'Informative'][nInformative:]==0), "Informative genes included in non-informative section")
        informative = allGenes.loc[:,'Gain']>allGenes.loc[:,'MDLP-Criterion']
        self.assertTrue(np.alltrue(informative==allGenes.loc[:,'Informative']), "Gain vs MDLP-criterion does not match marking as informative")

    def test_cuffnorm_multiclass_analysis(self):
        path = self._testFilePath('genes.count_table')
        approaches = {
            'CuffnormMultiClassCompare' : [0]*30 + [1]*29 + [2]*30 + [3]*30,
            'CuffnormOneVsRest' : [True]*30 + [False]*89
        }
        objs = dict()
        for approach, specimenClasses in approaches.items():
            objs[approach] = getattr(workflows, approach)(path, specimenClasses)

        for allGenes in [True, False]:
            output = dict()
            for approach, obj in objs.items():
                output[approach] = obj.informativeGenes(allGenes)
            mul = output['CuffnormMultiClassCompare']
            one = output['CuffnormOneVsRest']

            self.assertTrue(np.alltrue(mul[0]==one), "Data re informative genes does not match between multi-class and one-vs-rest approaches")
            self.assertTrue(np.alltrue(mul[0].index==one.index), "Ranking of genes does not match between multi-class and one-vs-rest approaches")
            self.assertEqual(len(mul.keys()), 4, "Multi-class informative gene analysis returns incorrect number of analyses for number of classes")

    def test_multiclass_to_onevsrest_class_conversion(self):
        multiClasses = [0]*7 + [1]*9 + [0]*3 + [2]*8
        expected = [
            [True]*7 + [False]*9 + [True]*3 + [False]*8,
            [False]*7 + [True]*9 + [False]*11,
            [False]*19 + [True]*8
        ]

        for trueClass, output in enumerate(expected):
            self.assertEqual(len(output), len(multiClasses), "Test vector for trueClass=%d has incorrect number of expected binary classes" % trueClass)
            binaryClasses = workflows.PandasMultiClassCompare._getBinaryClasses(multiClasses, trueClass)
            self.assertTrue(np.alltrue(output==binaryClasses), "Multi-class analyses do not properly convert to binary classes for trueClass=%d" % trueClass)

if __name__=='__main__':
    unittest.main()
