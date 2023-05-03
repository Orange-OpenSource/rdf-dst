import re
import numpy as np

def postprocess_rdfs(decoded_batch):
    """
    returns several rdf triplets per batch
    """

    regexSplit = re.compile(r"(?<!\s),(?!\s)")
    decoded_batch = [regexSplit.split(row) for row in decoded_batch]
    decoded_batch = [[word.strip() for word in rdfs] for rdfs in decoded_batch]
    clean_rdfs = [set([tuple(rdfs[i:i+3]) for i in range(0, len(rdfs), 3)]) for rdfs in decoded_batch]
    #linearized_rdfs = [['|'.join(rdf) for rdf in rdfs] for rdfs in clean_rdfs]
    return clean_rdfs
    #return {"clean_rdfs": clean_rdfs, "linearized_rdfs": linearized_rdfs}



class DSTMetrics:
    def __init__(self):
        self.turn_jga_scores = []

    def joint_goal_accuracy(self, predictions, references, index_dict):
        if (not predictions) and (not references):
            return sum(self.turn_jga_scores) / len(self.turn_jga_scores)

        batch_preds = predictions[0]
        batch_refs = references[0]
        scores = []
        #inv_index = {v: k for k, v in index_dict.items()}
        for pred_turn, ref_turn in zip(batch_preds, batch_refs):
            if len(ref_turn) == len(pred_turn):
                pred_turn_idx = [index_dict[rdf] for rdf in pred_turn if len(rdf) == 3]
                ref_turn_idx = [index_dict[rdf] for rdf in ref_turn if len(rdf) == 3]  # some slot values are not flawless rdfs in the annotated set
                missing_els_from_ref = set(pred_turn_idx) - set(ref_turn_idx)
                score = 0 if missing_els_from_ref else 1
                scores.append(score)
            else:
                scores.append(0)
        jga_turn_score = sum(scores) / len(scores) if len(scores) != 0 else 0
        self.turn_jga_scores.append(jga_turn_score)
        return self.joint_goal_accuracy(predictions[1:], references[1:], index_dict)
