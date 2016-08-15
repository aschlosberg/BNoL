import sys
import numpy as np, pandas as pd, grequests, shelve
import logging
from collections import deque
from . import information

class PandasInformativeGenes(object):
    """Binary comparison of sub-classes of specimens. In the case of multiple specimen classes, expected use is in a one-vs-rest fashion.

    Provides standardized access to means of ranking genes e.g. through entropy improvement via :py:class:`~bnol.information.Discretize`.

    Args:
        specimens (pandas.DataFrame): shape (n,p) where the n index values constitute the specimens and the p columns the genes.
        classes (numpy.ndarray): shape (n,) class designation for each of the n specimens.

    Raises:
        Exception: if there is not exactly one class designation for each specimen.
    """
    def __init__(self, specimens, classes):
        self.specimens = np.asarray(specimens)
        self.classes = classes

        if not self.specimens.shape[0]==len(self.classes):
            raise Exception("A single class designation must be provided for each specimen.")

        self.genes = specimens.columns.values
        self.specimenLabels = specimens.index
        self.Discrete = information.Discretize()
        logging.debug("PandasInformativeGenes initialized successfully")

    def _classOverExpressionRatio(self, discretizedFeatures, overExpressionClass):
        """Convenience wrapper for the mean number of times that each gene is over-expressed, i.e. marked True, in the subset of genes passed defined by booleanClasses==True."""
        inClass = [discretizedFeatures[idx] for (idx, c) in enumerate(self.classes) if c==overExpressionClass]
        return np.mean(inClass, axis=0)

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
        logging.debug("Starting feature discretization in PandasInformativeGenes")
        D.fit(self.specimens, self.classes)
        logging.debug("Finished feature discretization in PandasInformativeGenes")

        # determine a set of gene indices, ranked by decreasing entropy gain above MDLP criterion for said genes
        # the approach is not optimal in that we sort all p genes and then take a subset rather than taking the subset first
        # however, this is only run once, is very quick when tested with 22k+ genes and makes the code easier to read
        # simply sort and then take first p` where p` = p if allGenes else the total number of genes that are informative
        numGenesToInclude = len(self.genes) if allGenes else np.sum(self.Discrete.includeFeatures)
        gainAboveMDLP = self.Discrete.gains - self.Discrete.mdlpCriteria
        rankedGeneIndices = list(reversed(np.argsort(gainAboveMDLP)))[:numGenesToInclude]

        # for each of gene classes, calculate the proportion of specimens that show over-expression relative to the threshold
        (__, indices) = np.unique(self.classes, return_index=True)
        indices.sort()
        uniqueClasses = [self.classes[idx] for idx in indices] # note that this maintains the order as seen in the original set of classes

        orderedFeatures = D.discretizedFeatures[:,rankedGeneIndices]
        overExpression = [self._classOverExpressionRatio(orderedFeatures, c) for c in uniqueClasses]
        assert overExpression[0].shape==(len(rankedGeneIndices),), "Calculated over-expression ratio along incorrect axis"

        logging.debug("Included features determined, building DataFrame in PandasInformativeGenes")
        return pd.DataFrame(
            data = np.vstack((
                    D.includeFeatures[rankedGeneIndices],
                    D.gains[rankedGeneIndices],
                    D.mdlpCriteria[rankedGeneIndices],
                    D.bestThresholds[rankedGeneIndices],
                    np.vstack(tuple(overExpression))
                )).T,
            index=self.genes[rankedGeneIndices],
            columns=['Informative', 'Gain', 'MDLP-Criterion', 'Threshold'] + ["OverExpressed-%s" % c for c in uniqueClasses],
            dtype='float',
            copy=True,
        )

class CuffnormReader(object):
    """Class to abstract conversion of cuffnorm output to Pandas DataFrame."""
    @staticmethod
    def getSpecimens(cuffnormOutputPath):
        logging.info("Fetching cuffnorm output from: %s" % cuffnormOutputPath)
        return pd.read_csv(cuffnormOutputPath, delimiter='\t', index_col=0).T

class CuffnormInformativeGenes(PandasInformativeGenes, CuffnormReader):
    """Convenience wrapper for :py:class:`PandasInformativeGenes` to automatically load data from cuffnorm gene and FPKM counts.

    Args:
        cuffnormOutputPath (string): path to cuffnorm output
        classes (numpy.ndarray): shape (n,) defining categorical designation of each specimen present in cuffnormOutputPath
    """
    def __init__(self, cuffnormOutputPath, classes):
        specimens = self.getSpecimens(cuffnormOutputPath)
        logging.debug("CuffnormOveVsRest handing over to parent class PandasInformativeGenes")
        super(CuffnormInformativeGenes, self).__init__(specimens, classes)

class PandasMultiClass(PandasInformativeGenes):
    """Convenience wrapper to run :py:class:`PandasInformativeGenes` analyses on multi-class data by running c different one-vs-rest analyses, where c is the total number of classes.

    Args:
        specimens (pandas.DataFrame): shape (n,p) where the n index values constitute the specimens and the p columns the genes.
        multiClasses (list): length (n) class designation for each of the n specimens; can include any type that may be used as a dict key.
    """
    def __init__(self, specimens, multiClasses):
        self.multiClasses = multiClasses
        super(PandasMultiClass, self).__init__(specimens, multiClasses) # will modify classes for each run if doing oneVsRest, but need something in there for now and this will check the size of the array

    def informativeGenes(self, allGenes=False):
        """See :py:class:`bnol.workflows.PandasInformativeGenes.informativeGenes` for base behaviour, repeated for each class.

        Args:
            allGenes (bool): as it says on the box if True else only return those genes for which the entropy gain is greater than the MDLP criterion.

        Returns:
            dict: pandas.DataFrame: each entry being a pandas.DataFrame object returned by one-vs-rest analysis.
        """
        uniqueClasses = np.unique(self.multiClasses)
        allComparisons = dict()
        logging.info("Performing one-vs-rest multi-class comparison of %d classes in PandasMultiClass" % len(uniqueClasses))
        for c in uniqueClasses:
            logging.debug("Starting analysis of class: %s" % c)
            self.classes = np.asarray(self._getBinaryClasses(self.multiClasses, c))
            allComparisons[c] = super(PandasMultiClass, self).informativeGenes(allGenes)
            logging.debug("Finished analysis of class: %s" % c)
        return allComparisons

    @staticmethod
    def _getBinaryClasses(multiClasses, trueClass):
        notTrueClass = "not-%s" % trueClass
        return [(c if c==trueClass else notTrueClass) for c in multiClasses]

class CuffnormMultiClass(PandasMultiClass, CuffnormReader):
    """Convenience wrapper for :py:class:`PandasMultiClass` to automatically load data from cuffnorm gene and FPKM counts.

    Args:
        cuffnormOutputPath (string): path to cuffnorm output
        multiClasses (list): length (n) class designation for each of the n specimens present in cuffnormOutputPath; can include any type that may be used as a dict key.
    """
    def __init__(self, cuffnormOutputPath, multiClasses):
        specimens = self.getSpecimens(cuffnormOutputPath)
        super(CuffnormMultiClass, self).__init__(specimens, multiClasses)

ensemblFormats = {'full', 'condensed'} # set {} has faster lookup than list []
def EnsemblLookup(ensemblIDs, lookupFormat='full', rebuildCache=False):
    """Use the Ensembl REST API 'lookup' to return data corresponding to a particular ID. Will create a local cache.

    http://rest.ensembl.org/documentation/info/lookup

    Args:
        ensemblIDs (list or string): Ensembl IDs in a list; will accept a string.
        lookupFormat (string): One of 'full' or 'condensed' as described in the API documentation.
        rebuildCache (bool): If True, fetch fresh data from Ensembl even if a locally-cached version exists.

    Returns:
        list or dict: The JSON data returned by the API, converted to a dict. Will return a single dict if string ID passed.

    Raises:
        Exception: if an invalid format string is passed.
    """
    if not lookupFormat in ensemblFormats:
        raise Exception("Ensembl lookup can only be 'full' or 'condsensed'")

    if isinstance(ensemblIDs, str):
        ensemblIDs = [ensemblIDs]
        returnAsList = False
    else:
        returnAsList = True
        ensemblIDs = ensemblIDs

    memo = shelve.open("./.ensemblCache-%d" % sys.version_info[0]) # technically don't need different file names, but local automated testing fails when using python3 to open python2 shelves

    getFresh = [rebuildCache or (not idx in memo) or (memo[idx] is None) for idx in ensemblIDs]
    urls = ["https://rest.ensembl.org/lookup/id/%s?content-type=application/json;format=%s" % (idx, lookupFormat) for (i, idx) in enumerate(ensemblIDs) if getFresh[i]]
    rs = (grequests.get(u) for u in urls)
    freshData = deque(grequests.map(rs))

    data = []
    for i, idx in enumerate(ensemblIDs):
        if getFresh[i]:
            memo[idx] = freshData.popleft() # freshData queue is only populated with those we know are not in the memo (or all if rebuildCache)
        data.append(memo[idx])

    memo.close()

    def returnValue(d):
        return d.json() if (d is not None and d.status_code==200) else dict()

    if returnAsList:
        return [returnValue(d) for d in data]
    else:
        return returnValue(data[0])
