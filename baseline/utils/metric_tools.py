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
            self.data = {"preds": self.decoded_preds, "labels": self.decoded_labels}

        preds = self.data["preds"]
        labels = self.data["labels"]
        scores = self.f1_states(preds, labels)
        span_scores = self.span_evaluation(preds, labels)
        jga = self.joint_goal_accuracy(preds, labels)
        scores.update(jga)
        scores.update(span_scores)
        return scores


    def span_evaluation(self, preds, labels):
        preds = [';'.join(p) for p in preds]
        labels = [';'.join(l) for l in labels]
        meteor_score = self.meteor.compute(predictions=preds, references=labels)['meteor']  # getting the dumb value from dict object
        gleu_score = self.gleu.compute(predictions=preds, references=labels)['google_bleu']

        #span_score = np.mean([1 if p == l else 0 for p, l in zip(preds, labels)])
        return {"meteor": round(meteor_score, 2) * 100, "gleu": round(gleu_score, 2) * 100}
        #return {"meteor": round(meteor_score, 2) * 100, "gleu": round(gleu_score, 2) * 100, "span_accuracy": round(span_score, 2) * 100}

    
    def joint_goal_accuracy(self, preds, labels):

        joint_goal_accuracy_score = []
        for p, l in zip(preds, labels):
            score = []

            # more generous
            for state in l:
                score.append(1 if state in p else 0)

            # all hallucinations drop score to 0, that is why we use the more generous approach
            #for state in p:
            #    score.append(1 if state in l else 0)


            joint_goal_accuracy_score.append(1 if 0 not in score else 0)

        return {"jga": round(np.mean(joint_goal_accuracy_score), 2) * 100}

        #score = []
        #for p, l in zip(preds, labels):
        #    # more generous, otherwise we punish hallucinations even if all states were accurate predicted
        #    score.append(1 if l in preds else 0)
        #    #score.append(1 if p == l else 0)
        #    #if p not in labels:
        #    #    score.append(0)
        #    #else:
        #    #    score.append(1 if l in preds else 0)

        #return {"jga": round(np.mean(score), 2) * 100}

        # leo's is stricter
        #scores = []
        #for p, l in zip(preds, labels):
        #    if len(l) != 0:
        #        #if p == l:
        #        #if l in preds and (len(preds) == len(labels)):
        #        if l in preds:
        #            #print(f"This is pred:\n{p}\nthis is label:\n{l}")
        #            #print('\n'*2)
        #            #print(preds[preds.index(l)])
        #            #print()
        #            scores.append(1)
        #        elif l not in preds:
        #            scores.append(0)
        #        
        #        if p not in labels:
        #            scores.append(0)

        #    else:
        #        if len(p) == 0:
        #            scores.append(1)
        #        else:
        #            scores.append(0)

        #result =  sum(scores) / len(scores) if len(scores) != 0 else 0
        #return {"jga": round(result, 2) * 100}

    

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

        f1 = round(f1, 2) * 100
        p = round(p, 2) * 100
        r = round(r, 2) * 100
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