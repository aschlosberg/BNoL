from bnol import workflows
specimenClasses = ['TissueA']*30 + ['TissueB']*29 + ['TissueC']*30 + ['TissueD']*30
analysis = workflows.CuffnormInformativeGenes('tests/data/genes.count_table', specimenClasses)
genes = analysis.informativeGenes(allGenes=True)
print(genes)
