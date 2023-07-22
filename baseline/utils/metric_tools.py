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
            self.dialogue_inputs = [out['inputs'] for out in outputs]

            self.data = {"preds": self.decoded_preds, "labels": self.decoded_labels,
                         "inputs": self.dialogue_inputs, "ids": self.dialogue_ids}

        preds_batch = self.data["preds"]
        labels_batch = self.data["labels"]
        # flattening
        preds = [self.normalize(p) for preds in preds_batch for p in preds]
        labels = [self.normalize(l) for labels in labels_batch for l in labels]

        #self.diag_construction()
        #reordered_preds_batch = self.reordered_dialogues['preds']
        #reordered_labels_batch = self.reordered_dialogues['labels']
        #reordered_preds = [self.normalize(p) for preds in reordered_preds_batch for p in preds]
        #reordered_labels = [self.normalize(l) for labels in reordered_labels_batch for l in labels]

        span_scores = self.span_evaluation(preds, labels)
        ga_scores = self.goal_accuracy(preds, labels)
        f1_scores = self.f1_states(preds, labels)
        return {**span_scores, **ga_scores, **f1_scores}
    
    
    @staticmethod
    def normalize(states):
        return frozenset('='.join([slot, value]) for slot, value in states.items())
    #        #normalized = frozenset('='.join([slot, value]) for dictionary in batch for slot, value in dictionary.items() if slot != '_NONE_')

    def diag_construction(self):
        ids = self.data['ids']
        all_ids = [i for batch_ids in ids for i in batch_ids]

        unique_ids = {i: v for i, v in enumerate(set(all_ids))}
        inverse_ids = {v: k for k, v in unique_ids.items()}
        self.unique_ids = unique_ids
        self.inverse_ids = inverse_ids

        preds = self.data['preds']
        all_preds = [p for batch_preds in preds for p in batch_preds]
        labels = self.data['labels']
        all_labels = [l for batch_labels in labels for l in batch_labels]
        inputs = self.data['inputs']
        all_inputs = [l for batch_inputs in inputs for l in batch_inputs]

        new_dialogues = {}
        turn_number = 0

        for idx, p, l, ip in zip(all_ids, all_preds, all_labels, all_inputs):
            diag_id = inverse_ids[idx] 
            diag_data = [{'preds': p, 'labels': l, 'inputs': ip}]
            if diag_id in new_dialogues:
                turn_number += 1
                new_dialogues[diag_id].extend(diag_data)
            else:
                turn_number = 0
                new_dialogues[diag_id] = diag_data
        

        preds = []
        labels = []
        inputs = []
        for k in new_dialogues.keys():
            preds.append([data['preds'] for data in new_dialogues[k]])
            labels.append([data['labels'] for data in new_dialogues[k]])
            inputs.append([data['inputs'] for data in new_dialogues[k]])

        self.reordered_dialogues = {"preds": preds, "labels": labels, "inputs": inputs}
        

    def span_evaluation(self, preds, labels):
        preds = [';'.join(p) for p in preds]
        labels = [';'.join(l) for l in labels]
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
            flexible_match = l <= p

            fga.append(1 if flexible_match else 0)

            exact_match = p ^ l
            jga.append(1 if len(exact_match) == 0 else 0)

            fuzzy_jga.append(1 if fuzz.partial_ratio(p, l) >= fuzzy_ratio else 0)

        return {"jga": round(np.mean(jga), 5) * 100,
                "fga_exact_recall": round(np.mean(fga), 5) * 100,
                "fuzzy_jga": round(np.mean(fuzzy_jga), 5) * 100,
                }


    def f1_states(self, preds, labels):

        intersections = [
             len(c & r) for c, r in zip(preds, labels)
         ]

        #false positives
        false_pos = [
            len(c - r) for c, r in zip(preds, labels)
        ]

        #false negatives
        false_negs = [
            len(r - c) for r, c in zip(labels, preds)
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

        #predicted_set = set(preds)
        #true_set = set(labels)

        #tp = len(predicted_set & true_set)
        #fp = len(predicted_set - true_set)
        #fn = len(true_set - predicted_set)

        #precision = tp / (tp + fp) if tp + fp > 0 else 0
        #recall = tp / (tp + fn) if tp + fn > 0 else 0
        #f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        #return {'precision': round(precision, 2) * 100, 'recall': round(recall, 2) * 100, 'f1' : round(f1, 2) * 100}
