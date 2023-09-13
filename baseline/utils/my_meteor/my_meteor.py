# Copyright (c) 2023 Orange

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITEDTOTHE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Software Name : knowledge-graph-dst
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: H. Andres Gonzalez

""" METEOR metric. """

import numpy as np
from datasets.config import importlib_metadata, version
from nltk.translate import meteor_score


NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))
if NLTK_VERSION >= version.Version("3.6.4"):
    from nltk import word_tokenize


_CITATION = """\
@inproceedings{banarjee2005,
  title     = {{METEOR}: An Automatic Metric for {MT} Evaluation with Improved Correlation with Human Judgments},
  author    = {Banerjee, Satanjeev  and Lavie, Alon},
  booktitle = {Proceedings of the {ACL} Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization},
  month     = jun,
  year      = {2005},
  address   = {Ann Arbor, Michigan},
  publisher = {Association for Computational Linguistics},
  url       = {https://www.aclweb.org/anthology/W05-0909},
  pages     = {65--72},
}
"""

_DESCRIPTION = """\
METEOR, an automatic metric for machine translation evaluation
that is based on a generalized concept of unigram matching between the
machine-produced translation and human-produced reference translations.
Unigrams can be matched based on their surface forms, stemmed forms,
and meanings; furthermore, METEOR can be easily extended to include more
advanced matching strategies. Once all generalized unigram matches
between the two strings have been found, METEOR computes a score for
this matching using a combination of unigram-precision, unigram-recall, and
a measure of fragmentation that is designed to directly capture how
well-ordered the matched words in the machine translation are in relation
to the reference.

METEOR gets an R correlation value of 0.347 with human evaluation on the Arabic
data and 0.331 on the Chinese data. This is shown to be an improvement on
using simply unigram-precision, unigram-recall and their harmonic F1
combination.
"""

_KWARGS_DESCRIPTION = """
Computes METEOR score of translated segments against one or more references.
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    alpha: Parameter for controlling relative weights of precision and recall. default: 0.9
    beta: Parameter for controlling shape of penalty as a function of fragmentation. default: 3
    gamma: Relative weight assigned to fragmentation penalty. default: 0.5
Returns:
    'meteor': meteor score.
Examples:

    >>> meteor = evaluate.load('meteor')
    >>> predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    >>> references = ["It is a guide to action that ensures that the military will forever heed Party commands"]
    >>> results = meteor.compute(predictions=predictions, references=references)
    >>> print(round(results["meteor"], 4))
    0.6944
"""


class Meteor:

    def compute(self, predictions, references, alpha=0.9, beta=3, gamma=0.5):
        multiple_refs = isinstance(references[0], list)
        if NLTK_VERSION >= version.Version("3.6.5"):
            # the version of METEOR in NLTK version 3.6.5 and earlier expect tokenized inputs
            if multiple_refs:
                scores = [
                    meteor_score.meteor_score(
                        [word_tokenize(ref) for ref in refs],
                        word_tokenize(pred),
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                    )
                    for refs, pred in zip(references, predictions)
                ]
            else:
                scores = [
                    meteor_score.single_meteor_score(
                        word_tokenize(ref), word_tokenize(pred), alpha=alpha, beta=beta, gamma=gamma
                    )
                    for ref, pred in zip(references, predictions)
                ]
        else:
            if multiple_refs:
                scores = [
                    meteor_score.meteor_score(
                        [[word_tokenize(ref) for ref in group] for group in references][0],
                        word_tokenize(pred),
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                    )
                    for ref, pred in zip(references, predictions)
                ]
            else:
                scores = [
                    meteor_score.single_meteor_score(ref, pred, alpha=alpha, beta=beta, gamma=gamma)
                    for ref, pred in zip(references, predictions)
                ]

        return {"meteor": np.mean(scores)}
