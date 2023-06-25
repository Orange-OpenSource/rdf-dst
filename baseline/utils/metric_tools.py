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
        scores = self.f1_states(preds, labels)
        span_scores = self.span_evaluation(preds, labels)
        jga = self.joint_goal_accuracy(preds, labels)
        scores.update(jga)
        scores.update(span_scores)
        return scores


    def flatten_batches(self):
        flatten_labels = [state for label in self.decoded_labels for state in label]

        flatten_preds = [state for pred in self.decoded_preds for state in pred]
        return {"preds": flatten_preds, "labels": flatten_labels}
    
    def span_evaluation(self, preds, labels):
        preds = ';'.join(preds)
        labels = ';'.join(labels)
        meteor_score = self.meteor.compute(predictions=[preds], references=[labels])['meteor']  # getting the dumb value from dict object
        gleu_score = self.gleu.compute(predictions=[preds], references=[labels])['google_bleu']
        return {"meteor": round(meteor_score, 2) * 100, "gleu": round(gleu_score, 2) * 100}

    
    def joint_goal_accuracy(self, preds, labels):
        score = []
        for l in labels:
            score.append(1 if l in preds else 0)

        return {"jga": round(np.mean(score), 2) * 100}

        #scores = []
        #for p, l in zip(preds, labels):
        #    if len(l) != 0:
        #        #if p == l:
        #        if l in preds:
        #            print(f"This is pred:\n{p}\nthis is label:\n{l}")
        #            print('\n'*2)
        #            print(preds[preds.index(l)])
        #            print()
        #            scores.append(1)
        #        else:
        #            scores.append(0)
        #    else:
        #        if len(p) == 0:
        #            scores.append(1)
        #        else:
        #            scores.append(0)

        #result =  sum(scores) / len(scores) if len(scores) != 0 else 0
        #return {"jga": round(result, 2) * 100}

    

    def f1_states(self, newcandlist, newreflist):
        """
        """

        intersection = set(newcandlist) & set(newreflist)
        precision = len(intersection) / len(newcandlist) if len(newcandlist) > 0 else 1
        recall = len(intersection) / len(newreflist) if len(newreflist) > 0 else 1

        try:
            f1 = 2 * (precision*recall) / (precision+recall)
        except ZeroDivisionError:
            f1 = 0

        f1 = round(f1, 2) * 100
        p = round(precision, 2) * 100
        r = round(recall, 2) * 100
        return {'precision': p, 'recall': r, 'f1' : f1}

