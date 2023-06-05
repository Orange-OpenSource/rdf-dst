from collections import Counter
import evaluate
import numpy as np

class DSTMetrics:

    def __init__(self, outputs, from_file: bool=False):
        if from_file:
            self.data = outputs
        else:
            self.decoded_labels = [out['labels'] for out in outputs]
            self.decoded_preds = [out['preds'] for out in outputs]
            self.dialogue_ids = [out['ids'] for out in outputs]
            self.data = self.flatten_batches()

        self.slots_empty_assignment = ["none", '', ' ', '*']

        evaluate.load("squad")

    def __call__(self):
        preds = self.data["preds"]
        labels = self.data["labels"]
        jga = self.joint_goal_accuracy(preds, labels)
        rdf_f1 = self.exact_triple_scores(preds, labels)
        squad_f1 = self.counter_squad_method(preds, labels)
        scores = {"jga": jga}
        scores.update(squad_f1)
        scores.update(rdf_f1)
        return scores


    def flatten_batches(self):
        flatten_preds = [rdf for pred in self.decoded_preds for rdf in pred]
        flatten_labels = [rdf for label in self.decoded_labels for rdf in label]
        return {"preds": flatten_preds, "labels": flatten_labels}
    
    def joint_goal_accuracy(self, preds, labels):
        all_scores = []
        for p, l in zip(preds, labels):
            score = []
            #if len(p) == len(l):
            for rdf in l:
                score.append(1 if rdf in p else 0)
            #else:
            score.append(0)

            all_scores.append(np.mean(score))

        return round(np.mean(all_scores), 2) * 100
    
    def f1_score(self, pred, ref):
        ref = "|".join(ref).split('|')
        pred = "|".join(pred).split('|')
        common = Counter(ref) & Counter(pred)
        num_same = sum(common.values())

        if num_same == 0:
            return {'precision': 0, 'recall': 0, 'f1' : 0}

        precision = 1.0 * num_same / len(pred)
        recall = 1.0 * num_same / len(ref)
        f1 = (2 * precision * recall) / (precision + recall)

        return {'precision': precision, 'recall': recall, 'f1' : f1}


    def counter_squad_method(self, candlist, reflist):
        # exact match is already JGA, so no need for implementing exact match here
        # this is looking for the nodes, much more flexible
        """
        closer to UAS
        """
        f1 = []
        precision = []
        recall = []
        for ref, pred in zip(reflist, candlist):
            res = self.f1_score(ref, pred)
            f1.append(res['f1'])
            precision.append(res['precision'])
            recall.append(res['recall'])

        f1 = sum(f1) / len(f1)
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)

        f1 = round(f1, 2) * 100
        precision = round(precision, 2) * 100
        recall = round(recall, 2) * 100
        return {'squad_precision': precision, 'squad_recall': recall, 'squad_f1' : f1}


    def exact_triple_scores(self, newcandlist, newreflist):
        """
        closer LAS, smatch
        """

        intersections = [
            set(c) & set(r) for c, r in zip(newcandlist, newreflist)
        ]


        precisions = [
            len(i) / len(c) if len(c) > 0 else 1
            for i,c in zip(intersections, newcandlist)
        ]

        recalls = [
            len(i) /len(r) if len(r) > 0 else 1
            for i,r in zip(intersections, newreflist)
        ]

        p = sum(precisions) / len(precisions)
        r = sum(recalls) / len(recalls)
        try:
            f1 = 2 * (p*r) / (p+r)
        except ZeroDivisionError:
            f1 = 0

        f1 = round(f1, 2) * 100
        p = round(p, 2) * 100
        r = round(r, 2) * 100
        return {'precision': p, 'recall': r, 'f1' : f1}

