import numpy as np
import evaluate

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
            self.data = self.flatten_batches()

        preds = self.data["preds"]
        labels = self.data["labels"]
        scores = self.f1_smatch(preds, labels)
        span_scores = self.span_evaluation(preds, labels)
        jga = self.joint_goal_accuracy(preds, labels)
        scores.update(jga)
        scores.update(span_scores)
        return scores


    def flatten_batches(self):
        flatten_preds = [rdf for pred in self.decoded_preds for rdf in pred]
        flatten_labels = [rdf for label in self.decoded_labels for rdf in label]
        return {"preds": flatten_preds, "labels": flatten_labels}
    
    def span_evaluation(self, preds, labels):
        preds = [','.join(p) for p in preds]
        labels = [','.join(p) for p in labels]
        meteor_score = self.meteor.compute(predictions=preds, references=labels)['meteor']  # getting the dumb value from dict object
        gleu_score = self.gleu.compute(predictions=preds, references=labels)['google_bleu']

        #span_score = np.mean([1 if p == l else 0 for p, l in zip(preds, labels)])
        #return {"meteor": round(meteor_score, 2) * 100, "gleu": round(gleu_score, 2) * 100, "span_accuracy": round(span_score, 2) * 100}

        return {"meteor": round(meteor_score, 5) * 100, "gleu": round(gleu_score, 5) * 100}

    
    def joint_goal_accuracy(self, preds, labels):

        joint_goal_accuracy_score = []
        aga_score = []
        for p, l in zip(preds, labels):
            score = []

            # more generous
            for rdf in l:
                score.append(1 if rdf in p else 0)
            
            # stricter
            joint_goal_accuracy_score.append(1 if p == l else 0)


            aga_score.append(1 if 0 not in score else 0)

        return {"jga": round(np.mean(joint_goal_accuracy_score), 5) * 100, "aga": round(np.mean(aga_score), 5) * 100}

    def f1_smatch(self, newcandlist, newreflist):
        """
        closer LAS, smatch
        """


        intersections = [
            len(set(c) & set(r)) for c, r in zip(newcandlist, newreflist)
        ]

        #false positives
        false_pos = [
            len(set(c) - set(r)) for c, r in zip(newcandlist, newreflist)
        ]

        #false negatives
        false_negs = [
            len(set(r) - set(c)) for r, c in zip(newreflist, newcandlist)
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

