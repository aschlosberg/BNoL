import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from bnol import workflows
import unittest
from numpyfied import NumpyfiedTestCase
import numpy as np

class WorkflowsTester(NumpyfiedTestCase):

    @staticmethod
    def _testFilePath(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', filename)

    def test_cuffnorm_multiclass_analysis(self):
        path = self._testFilePath('genes.count_table')
        approaches = {
            'CuffnormMultiClassCompare' : [0]*30 + [1]*29 + [2]*30 + [3]*30,
            'CuffnormOneVsRest' : [True]*30 + [False]*89
        }
        objs = dict()
        for approach, specimenClasses in approaches.items():
            objs[approach] = getattr(workflows, approach)(path, specimenClasses)

        # Compare multiclass to one-vs-rest
        allOutput = dict()
        for allGenes in [True, False]:
            output = dict()
            for approach, obj in objs.items():
                output[approach] = obj.informativeGenes(allGenes)
            allOutput[str(allGenes)] = output
            mul = output['CuffnormMultiClassCompare']
            one = output['CuffnormOneVsRest']

            self.assertAllTrue(mul[0]==one, "Data re informative genes does not match between multi-class and one-vs-rest approaches (allGenes=%s)" % allGenes)
            self.assertAllTrue(mul[0].index==one.index, "Ranking of genes does not match between multi-class and one-vs-rest approaches (allGenes=%s)" % allGenes)
            self.assertEqual(len(mul.keys()), 4, "Multi-class informative gene analysis returns incorrect number of analyses for number of classes (allGenes=%s)" % allGenes)

        # Ensure that one-vs-rest has expected outcomes
        onlyInformative = allOutput['False']['CuffnormOneVsRest']
        allGenes = allOutput['True']['CuffnormOneVsRest']

        nInformative = len(onlyInformative.index)
        ranking = ["GENE_%s" % g for g in ['F','D','G','C','I','H','E','A','B']]

        self.assertAllTrue(allGenes.index==ranking, "Incorrect ranking of genes by entropy gain")
        self.assertAllTrue(onlyInformative.index==allGenes.index[:nInformative], "Incorrect subset of genes in informative grouping")
        self.assertAllTrue(onlyInformative==allGenes.iloc[:nInformative,:], "Values for informative genes do not match across informative / all-genes groups")
        self.assertAllTrue(onlyInformative.loc[:,'Informative']==True, "Non-informative genes included in informative-only group")
        self.assertAllTrue(allGenes.loc[:,'Informative'][nInformative:]==0, "Informative genes included in non-informative section")
        informative = allGenes.loc[:,'Gain']>allGenes.loc[:,'MDLP-Criterion']
        self.assertAllTrue(informative==allGenes.loc[:,'Informative'], "Gain vs MDLP-criterion does not match marking as informative")

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
            self.assertAllTrue(output==binaryClasses, "Multi-class analyses do not properly convert to binary classes for trueClass=%d" % trueClass)

if __name__=='__main__':
    unittest.main()
