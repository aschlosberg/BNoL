.. BNoL documentation master file, created by
   sphinx-quickstart on Sun Apr 17 00:01:56 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BNoL
====

`Source code available on GitHub <https://github.com/aschlosberg/BNoL>`_

.. |TravisBuildStatus| image:: https://travis-ci.org/aschlosberg/BNoL.svg?branch=master
    :alt: Build Status
.. _TravisBuildStatus: https://travis-ci.org/aschlosberg/BNoL

BNoL (*pron. bee-noll*) is a Python library for discrete feature selection with a primary focus on transcriptomic (gene-expression) analysis. Within transcriptomics there is a need for discovery of specific features (biomarkers) rather than weighted combinations as can be found through feature extraction methods such as PCA.

Although documented and tested, the code is very much in its infancy and function interfaces are open to change. Unit tests are currently focussed on sanity (e.g. dimensions of matrices / vectors) and very simple hand-worked expected results.

BNoL stands for *Bare Necessities of Life*, inspired by the line in the Jungle Book song: "Old Mother Nature's recipes; that bring the bare necessities of life".

.. toctree::
   :maxdepth: 2

   bnol

Example Usage
-------------

Differential gene expression can be determined optimally via `information-theoretic <https://en.wikipedia.org/wiki/Information_theory>`_ means with :py:class:`bnol.information.Discretize`
which, for every gene, determines the expression threshold that minimises the `entropy <https://en.wikipedia.org/wiki/Information_theory#Entropy_of_an_information_source>`_ of specimen *classes*
above and below the threshold. Classes are categorical and may take two or more values; e.g. cancer vs normal, or treatment vs control. A threshold that perfectly segregates classes will result
in zero entropy.

Automated workflows ease the process of analysis by, for example, importing directly from `Cufflinks <https://cole-trapnell-lab.github.io/cufflinks/>`_ as below. Note that the same workflows
can be used for any ordinal data by directly utilising :py:class:`bnol.workflows.PandasInformativeGenes`.

.. literalinclude:: ../../examples/cuffnorm.py
   :language: python
   :linenos:

The decrease in entropy, or the *gain* as described by Fayyad and Irani[1], is used to determine whether or not each gene is considered informative as per the `mimimum-description length <https://en.wikipedia.org/wiki/Minimum_description_length>`_
principle which is a formalised *Occam's razor*. Output columns include the entropy *Gain*, determined at the optimal *Threshold* that defines the cutoff for classifcation as over- or under-expression. The gain must exceed
the *MDLP-Criterion* [1] in order to be considered *Informative*. Output is ranked by (Gain - MDLP-Criterion), in descending order, and entropy values are base-2 (bits). For each category included in *classes*, the proportion
of specimens that exceed the expression threshold for the gene in question is also calculated.

.. literalinclude:: ../../examples/cuffnorm-output.txt

[1] `Fayyad and Irani (1993). Multi-interval discretization of continuous-valued attributes for classification learning. <http://ourmine.googlecode.com/svn-history/r1513/trunk/share/pdf/fayyad93.pdf>`_

Compatibility
-------------

Continuous integration is performed against the following versions of Python (`See BNoL testing results <https://travis-ci.org/aschlosberg/BNoL>`_):

- 2.7
- 3.3
- 3.4
- 3.5

Acknowledgements
----------------

This work was made possible by funding from, in alphabetical order:

* `Cambridge Trust <https://www.cambridgetrust.org/>`_

  - Cambridge Trust Scholarship

* `The Royal College of Pathologists of Australasia <https://www.rcpa.edu.au/About/RCPA-Foundation/Grants-and-Awards>`_

  - RCPA Foundation Mike and Carole Ralston Travelling Fellowship

* `The University of Sydney Travelling Scholarships <https://sydney.edu.au/scholarships/research/travelling_scholarships_main.shtml#travel>`_

  - Charles Gilbert Heydon Travelling Fellowship in Biological Sciences
  - Eleanor Sophia Wood Postgraduate Research Travelling Scholarship
