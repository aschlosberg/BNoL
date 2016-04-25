import numpy as np, pandas as pd
from . import information

class PandasOneVsRest(object):
    """Binary comparison of sub-classes of specimens. In the case of multiple specimen classes, expected use is in a one-vs-rest fashion.

    Provides standardized access to means of ranking genes e.g. through entropy improvement via :py:class:`~bnol.information.Discretize`.

    Args:
        specimens (pandas.DataFrame): shape (n,p) where the n index values constitute the specimens and the p columns the genes
        binaryClasses (numpy.ndarray): bool; shape (n,) class designation for each of the n specimens
    """
    def __init__(self, specimens, binaryClasses):
        self.specimens = np.asarray(specimens)
        self.binaryClasses = np.asarray(binaryClasses, dtype='bool')

        if not self.specimens.shape[0]==self.binaryClasses.shape[0]:
            raise Exception("A single class designation must be provided for each specimen.")

        self.genes = specimens.columns.values
        self.specimenLabels = specimens.index
        self.Discrete = information.Discretize()

    def _classOverExpressionRatio(self, discretizedFeatures, booleanClasses):
        """Convenience wrapper for the mean number of times that each gene is over-expressed, i.e. marked True, in the subset of genes passed defined by booleanClasses==True."""
        return np.mean(discretizedFeatures[booleanClasses], axis=0)

    def informativeGenes(self, allGenes=False):
        """Determine genes considered to be informative by means of decreasing class entropy through. Convenience wrapper for :py:class:`~bnol.information.Discretize`.

        Will additionally provide an optimal threshold for determining over- vs under-expression between the two classes.
        The proportion of specimens in each class, considered over-expressed by this threshold, will also be provided.

        Args:
            allGenes (bool): as it says on the box if True else only return those genes for which the entropy gain is greater than the MDLP criterion.

        Returns:
            pandas.DataFrame: details regarding genes, ranked in descending order of amount for which entropy gain exceeds MDLP criterion.
        """
        D = self.Discrete # convenience
        D.fit(self.specimens, self.binaryClasses)

        # determine a set of gene indices, ranked by decreasing entropy gain above MDLP criterion for said genes
        # the approach is not optimal in that we sort all p genes and then take a subset rather than taking the subset first
        # however, this is only run once, is very quick when tested with 22k+ genes and makes the code easier to read
        # simply sort and then take first p` where p` = p if allGenes else the total number of genes that are informative
        numGenesToInclude = len(self.genes) if allGenes else np.sum(self.Discrete.includeFeatures)
        gainAboveMDLP = self.Discrete.gains - self.Discrete.mdlpCriteria
        rankedGeneIndices = list(reversed(np.argsort(gainAboveMDLP)))[:numGenesToInclude]

        # for each of the True- and False-classed genes, calculate the proportion of specimens that show over-expression relative to the threshold
        overExpression = [self._classOverExpressionRatio(D.discretizedFeatures[:,rankedGeneIndices], c) for c in [D.booleanClasses, ~D.booleanClasses]]
        assert overExpression[0].shape==(len(rankedGeneIndices),), "Calculated over-expression ratio along incorrect axis"
        assert overExpression[0].shape==overExpression[1].shape, "Over- / under-expression ratios should have same shape"

        return pd.DataFrame(
            data = np.vstack((
                    D.includeFeatures[rankedGeneIndices],
                    D.gains[rankedGeneIndices],
                    D.mdlpCriteria[rankedGeneIndices],
                    D.bestThresholds[rankedGeneIndices],
                    overExpression[0],
                    overExpression[1],
                )).T,
            index=self.genes[rankedGeneIndices],
            columns=['Informative', 'Gain', 'MDLP-Criterion', 'Threshold', 'OverExpressedInTrue', 'OverExpressedInFalse'],
            dtype='float',
            copy=True,
        )

class CuffnormOneVsRest(PandasOneVsRest):
    """Convenience wrapper for :py:class:`PandasOneVsRest` to automatically load data from cuffnorm gene and FPKM counts.

    Args:
        cuffnormOutputPath (string): path to cuffnorm output
        binaryClasses (numpy.ndarray): bool; shape (n,); classification of each of the n specimens present in cuffnormOutputPath
    """
    def __init__(self, cuffnormOutputPath, binaryClasses):
        specimens = pd.read_csv(cuffnormOutputPath, delimiter='\t', index_col=0).T
        super(CuffnormOneVsRest, self).__init__(specimens, binaryClasses)
