import numpy as np
import evaluate
from fuzzywuzzy import fuzz
from utils.my_google_bleu.my_google_bleu import GoogleBleu
from utils.my_meteor.my_meteor import Meteor

import nltk

nltk.download("punkt")
nltk.download("wordnet")

# Create a downloader instance
#downloader = nltk.downloader.Downloader()
#
## Disable network access
##downloader._download = downloader._download_noop
#
#downloader.download_dir = '../metrics/nltk_data'
#
## Set the downloader for NLTK
#nltk.downloader.downloader = downloader

# Now NLTK will only use locally available data
import logging
logging.basicConfig(level=logging.INFO)


class DSTMetrics:

    def __init__(self):
        # may need to directly download punkt when running  from clusters and omw and wordnet?
        #self.gleu = evaluate.load("google_bleu")
        #self.meteor = evaluate.load("meteor")
        self.gleu = GoogleBleu()
        self.meteor = Meteor()
    

    def __call__(self, outputs, from_file: bool=False):
        if from_file:
            self.data = outputs
        else:
            self.decoded_labels = [out['labels'] for out in outputs]
            self.decoded_preds = [out['preds'] for out in outputs]
            self.dialogue_ids = [out['ids'] for out in outputs]
            self.data = self.flatten_batches()

        preds = self.data["preds"]
        labels = self.data["labels"]
        span_scores = self.span_evaluation(preds, labels)
        preds = [set(p) for p in preds]
        labels = [set(l) for l in labels]
        f1_scores = self.f1_smatch(preds, labels)
        ga = self.goal_accuracy(preds, labels)
        return {**span_scores, **ga, **f1_scores}


    def flatten_batches(self):
        flatten_preds = [rdf for pred in self.decoded_preds for rdf in pred]
        flatten_labels = [rdf for label in self.decoded_labels for rdf in label]
        return {"preds": flatten_preds, "labels": flatten_labels}
    
    def span_evaluation(self, preds, labels):
        preds = [','.join(p) for p in preds]
        labels = [','.join(p) for p in labels]
        meteor_score = self.meteor.compute(predictions=preds, references=labels)['meteor']  # getting the dumb value from dict object
        gleu_score = self.gleu.compute(predictions=preds, references=labels)['google_bleu']

        return {"meteor": round(meteor_score, 5) * 100, "gleu": round(gleu_score, 5) * 100}

    
    def goal_accuracy(self, preds, labels):

        # fuzzy is some sort of levensthein
        fuzzy_ratio = 95
        jga = []
        fga = []
        fuzzy_jga = []
        for p, l in zip(preds, labels):
            # more generous: is l a subset of p
            flexible_match = l <= p
            
            fga.append(1 if flexible_match else 0)
            # stricter, using symmetric difference
            exact_match = p ^ l
            jga.append(1 if len(exact_match) == 0 else 0)
            fuzzy_jga.append(1 if fuzz.partial_ratio(p, l) >= fuzzy_ratio else 0)

        return {"jga": round(np.mean(jga), 5) * 100,
                "fga_exact_recall": round(np.mean(fga), 5) * 100,
                "fuzzy_jga": round(np.mean(fuzzy_jga), 5) * 100,
                }

    def f1_smatch(self, newcandlist, newreflist):
        """
        closer LAS, smatch
        """


        intersections = [
            len(c & r) for c, r in zip(newcandlist, newreflist)
        ]

        #false positives
        false_pos = [
            len(c - r) for c, r in zip(newcandlist, newreflist)
        ]

        #false negatives
        false_negs = [
            len(r - c) for r, c in zip(newreflist, newcandlist)
        ]



        precisions = [
            i / (i + fp) if (i + fp) > 0 else 0
            for i, fp in zip(intersections, false_pos)
        ]


        recalls = [
            i / (i + fn) if (i + fn) > 0 else 0
            for i, fn in zip(intersections, false_negs)
        ]

        p = sum(precisions) / len(precisions)
        r = sum(recalls) / len(recalls)
        try:
            f1 = 2 * (p*r) / (p+r)
        except ZeroDivisionError:
            f1 = 0

        f1 = round(f1, 5) * 100
        p = round(p, 5) * 100
        r = round(r, 5) * 100
        return {'precision': p, 'recall': r, 'f1' : f1}

