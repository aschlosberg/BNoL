import numpy as np, pandas as pd
from . import information

class PandasOneVsRest(object):
    def __init__(self, specimens, binaryClasses):
        self.genes = specimens.columns.values
        self.specimenLabels = specimens.index

        self.Discrete = information.Discretize()
        self.Discrete.fit(np.asarray(specimens), np.asarray(binaryClasses, dtype='bool'))

    def _includeFeatures(self, allGenes=False):
        return [True]*len(self.genes) if allGenes else self.Discrete.includeFeatures

    def informativeGenes(self, allGenes=False):
        order = list(reversed(np.argsort(self.Discrete.gains)))

        D = self.Discrete
        return pd.DataFrame(
            data = np.vstack((
                    D.includeFeatures[order],
                    D.gains[order],
                    D.mdlpCriteria[order],
                )),
            columns=self.genes[order],
            index=['Informative', 'Gain', 'MDLP-Criterion'],
            dtype='float',
            copy=True,
        )


class CuffnormOneVsRest(PandasOneVsRest):
    def __init__(self, cufflinksOutputPath, binaryClasses):
        specimens = pd.read_csv(cufflinksOutputPath, delimiter='\t', index_col=0).T
        super(CuffnormOneVsRest, self).__init__(specimens, binaryClasses)
