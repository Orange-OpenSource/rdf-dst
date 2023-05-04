def index_encoding(preds, labels):
    """
    encodes all rdfs from preds and labels to vectorize evaluation 
    """
    pred_rdfs = [rdfs for batch in preds for rdfs in batch]
    pred_rdfs = set().union(*pred_rdfs)
    label_rdfs = [rdfs for batch in labels for rdfs in batch]
    label_rdfs = set().union(*label_rdfs)
    unique_rdfs = list(label_rdfs | pred_rdfs)
    invalid_val = 0
    rdf_vocab = dict()
    for i in range(len(unique_rdfs)):
        if len(unique_rdfs[i]) != 3:
            invalid_val -= 1
            rdf_vocab.setdefault(unique_rdfs[i], -i)
        else:
            rdf_vocab.setdefault(unique_rdfs[i], i)
    return rdf_vocab



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
