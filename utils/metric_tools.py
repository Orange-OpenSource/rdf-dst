import numpy as np
import evaluate

class DSTMetrics:

    def __init__(self):
        self.slots_empty_assignment = ["none", '', ' ', '*']
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
        return {"meteor": round(meteor_score, 2) * 100, "gleu": round(gleu_score, 2) * 100}

    
    def joint_goal_accuracy(self, preds, labels):
        joint_goal_accuracy = []
        for p, l in zip(preds, labels):
            score = []
            #print(f"References:\n{l}\nSize:{len(l)}")
            #print()
            #print(f"Predictions:\n{p}\nSize:{len(p)}")
            for rdf in l:
                score.append(1 if rdf in p else 0)
                #TODO: Consider an exact match and hallucinations match, exact will yield a really low score!

            joint_goal_accuracy.append(1 if 0 not in score else 0)

        return {"jga": round(np.mean(joint_goal_accuracy), 2) * 100}
    

    def f1_smatch(self, newcandlist, newreflist):
        """
        closer LAS, smatch
        """

        intersections = [
            set(c) & set(r) for c, r in zip(newcandlist, newreflist)
        ]


        # which one is which?
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

