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
        return self.genes[order]


class CufflinksOneVsRest(PandasOneVsRest):
    def __init__(self, cufflinksOutputPath, binaryClasses):
        specimens = pd.read_csv(cufflinksOutputPath, delimiter='\t', index_col=0).T
        super(CufflinksOneVsRest, self).__init__(specimens, binaryClasses)
