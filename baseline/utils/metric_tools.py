import numpy as np
import evaluate
from fuzzywuzzy import fuzz

class DSTMetrics:

    def __init__(self):
        # may need to directly download punkt when running  from clusters and omw and wordnet?
        self.gleu = evaluate.load("google_bleu")
        self.meteor = evaluate.load("meteor")
    

    def __call__(self, outputs, from_file: bool=False):
        if from_file:
            self.data = outputs
        else:
            self.decoded_labels = [out['labels'] for out in outputs]
            self.decoded_preds = [out['preds'] for out in outputs]
            self.dialogue_ids = [out['ids'] for out in outputs]

            self.data = {"preds": self.decoded_preds, "labels": self.decoded_labels}

        preds = self.flatten(self.data["preds"])
        labels = self.flatten(self.data["labels"])

        ga = self.goal_accuracy(preds, labels)
        f1_scores = self.f1_states(preds, labels)
        span_scores = self.span_evaluation(preds, labels)
        return {**span_scores, **ga, **f1_scores}


    def flatten(self, batch_slot_vals):
        if isinstance(batch_slot_vals[0], list):
            return [slot_vals for s_v in batch_slot_vals for slot_vals in s_v]
        return batch_slot_vals

    def span_evaluation(self, preds, labels):
        preds = [';'.join(p) for p in preds]
        labels = [';'.join(l) for l in labels]
        meteor_score = self.meteor.compute(predictions=preds, references=labels)['meteor']  # getting the dumb value from dict object
        gleu_score = self.gleu.compute(predictions=preds, references=labels)['google_bleu']

        return {"meteor": round(meteor_score, 5) * 100, "gleu": round(gleu_score, 5) * 100}

    
    def goal_accuracy(self, preds, labels):

        joint_goal_accuracy_score = []
        fga_score = []
        fuzz_jga = []
        # fuzzy is some sort of levensthein
        # TODO remove fuzzy ratio condition to see general fuzz 
        fuzzy_ratio = 95
        for p, l in zip(preds, labels):
            # more generous: is l a subset of p
            flexible_match = l <= p
            
            fga_score.append(1 if flexible_match else 0)
            # stricter, using symmetric difference
            exact_match = p ^ l
            joint_goal_accuracy_score.append(1 if len(exact_match) == 0 else 0)

        # no need to convert to string...
        #p = {key: value for slot_val in p for key, value in [slot_val.split('=')]}
        #l = {key: value for slot_val in l for key, value in [slot_val.split('=')]}
            fuzz_jga.append(1 if fuzz.partial_ratio(p, l) >= fuzzy_ratio else 0)

        return {"jga": round(np.mean(joint_goal_accuracy_score), 5) * 100,
                "fga_exact_recall": round(np.mean(fga_score), 5) * 100,
                "fuzzy_jga": round(np.mean(fuzz_jga), 5) * 100,
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
