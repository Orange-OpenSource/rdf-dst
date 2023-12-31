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

""" Google BLEU (aka GLEU) metric. """

from typing import Dict, List

import datasets
#from my_google_bleu.my_gleu_score import gleu_score
from nltk.translate import gleu_score

from .tokenizer_13a import Tokenizer13a


_CITATION = """\
@misc{wu2016googles,
      title={Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation},
      author={Yonghui Wu and Mike Schuster and Zhifeng Chen and Quoc V. Le and Mohammad Norouzi and Wolfgang Macherey
              and Maxim Krikun and Yuan Cao and Qin Gao and Klaus Macherey and Jeff Klingner and Apurva Shah and Melvin
              Johnson and Xiaobing Liu and Łukasz Kaiser and Stephan Gouws and Yoshikiyo Kato and Taku Kudo and Hideto
              Kazawa and Keith Stevens and George Kurian and Nishant Patil and Wei Wang and Cliff Young and
              Jason Smith and Jason Riesa and Alex Rudnick and Oriol Vinyals and Greg Corrado and Macduff Hughes
              and Jeffrey Dean},
      year={2016},
      eprint={1609.08144},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
The BLEU score has some undesirable properties when used for single
sentences, as it was designed to be a corpus measure. We therefore
use a slightly different score for our RL experiments which we call
the 'GLEU score'. For the GLEU score, we record all sub-sequences of
1, 2, 3 or 4 tokens in output and target sequence (n-grams). We then
compute a recall, which is the ratio of the number of matching n-grams
to the number of total n-grams in the target (ground truth) sequence,
and a precision, which is the ratio of the number of matching n-grams
to the number of total n-grams in the generated output sequence. Then
GLEU score is simply the minimum of recall and precision. This GLEU
score's range is always between 0 (no matches) and 1 (all match) and
it is symmetrical when switching output and target. According to
our experiments, GLEU score correlates quite well with the BLEU
metric on a corpus level but does not have its drawbacks for our per
sentence reward objective.
"""

_KWARGS_DESCRIPTION = """\
Computes corpus-level Google BLEU (GLEU) score of translated segments against one or more references.
Instead of averaging the sentence level GLEU scores (i.e. macro-average precision), Wu et al. (2016) sum up the matching
tokens and the max of hypothesis and reference tokens for each sentence, then compute using the aggregate values.

Args:
    predictions (list of str): list of translations to score.
    references (list of list of str): list of lists of references for each translation.
    tokenizer : approach used for tokenizing `predictions` and `references`.
        The default tokenizer is `tokenizer_13a`, a minimal tokenization approach that is equivalent to `mteval-v13a`, used by WMT.
        This can be replaced by any function that takes a string as input and returns a list of tokens as output.
    min_len (int): The minimum order of n-gram this function should extract. Defaults to 1.
    max_len (int): The maximum order of n-gram this function should extract. Defaults to 4.

Returns:
    'google_bleu': google_bleu score

Examples:
    Example 1:
        >>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', \
        'he read the book because he was interested in world history']
        >>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat'], \
        ['he was interested in world history because he read the book']]
        >>> google_bleu = evaluate.load("google_bleu")
        >>> results = google_bleu.compute(predictions=predictions, references=references)
        >>> print(round(results["google_bleu"], 2))
        0.44

    Example 2:
        >>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', \
        'he read the book because he was interested in world history']
        >>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat', \
        'It is a guide to action that ensures that the rubber duck will never heed the cat commands', \
        'It is the practical guide for the rubber duck army never to heed the directions of the cat'], \
        ['he was interested in world history because he read the book']]
        >>> google_bleu = evaluate.load("google_bleu")
        >>> results = google_bleu.compute(predictions=predictions, references=references)
        >>> print(round(results["google_bleu"], 2))
        0.61

    Example 3:
        >>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', \
        'he read the book because he was interested in world history']
        >>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat', \
        'It is a guide to action that ensures that the rubber duck will never heed the cat commands', \
        'It is the practical guide for the rubber duck army never to heed the directions of the cat'], \
        ['he was interested in world history because he read the book']]
        >>> google_bleu = evaluate.load("google_bleu")
        >>> results = google_bleu.compute(predictions=predictions, references=references, min_len=2)
        >>> print(round(results["google_bleu"], 2))
        0.53

    Example 4:
        >>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', \
        'he read the book because he was interested in world history']
        >>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat', \
        'It is a guide to action that ensures that the rubber duck will never heed the cat commands', \
        'It is the practical guide for the rubber duck army never to heed the directions of the cat'], \
        ['he was interested in world history because he read the book']]
        >>> google_bleu = evaluate.load("google_bleu")
        >>> results = google_bleu.compute(predictions=predictions,references=references, min_len=2, max_len=6)
        >>> print(round(results["google_bleu"], 2))
        0.4
"""


class GoogleBleu:

    def compute(
        self,
        predictions: List[str],
        references: List[List[str]],
        tokenizer=Tokenizer13a(),
        min_len: int = 1,
        max_len: int = 4,
    ) -> Dict[str, float]:
        # if only one reference is provided make sure we still use list of lists
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        references = [[tokenizer(r) for r in ref] for ref in references]
        predictions = [tokenizer(p) for p in predictions]
        return {
            "google_bleu": gleu_score.corpus_gleu(
                list_of_references=references, hypotheses=predictions, min_len=min_len, max_len=max_len
            )
        }
