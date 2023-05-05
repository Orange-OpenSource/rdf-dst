import pandas as pd

class DSTMetrics:

    def __init__(self, outputs):
        self.decoded_labels = [out['labels'] for out in outputs]
        self.decoded_preds = [out['preds'] for out in outputs]
        self.dialogue_ids = [out['ids'] for out in outputs]

    def compute(self, store=False):

        self.index_encoding()
        self.dialogue_reconstruction()

        self.turn_jga_scores = []
        no_context_jga = self.joint_goal_accuracy(self.decoded_preds, self.decoded_labels)
        no_context_jga = round(no_context_jga * 100, 3)
        # reinit the list for recursion eval
        self.turn_jga_scores = []
        context_jga = self.joint_goal_accuracy(self.ordered_dialogues["ordered_preds"], self.ordered_dialogues["ordered_labels"])
        context_jga = round(context_jga * 100, 3)
        self.decoded_preds.clear()
        self.decoded_labels.clear()
        self.dialogue_ids.clear()
        if store:
            self.store_model_predictions()
        self.ordered_dialogues.clear()
        return {"contextual_jga": context_jga, "no_contextual_jga": no_context_jga}

    def joint_goal_accuracy(self, predictions, references):
        if (not predictions) and (not references):
            return sum(self.turn_jga_scores) / len(self.turn_jga_scores)

        batch_preds = predictions[0]
        batch_refs = references[0]
        scores = []
        #inv_index = {v: k for k, v in self.rdf_indexes.items()}
        for pred_turn, ref_turn in zip(batch_preds, batch_refs):
            if len(ref_turn) == len(pred_turn):
                pred_turn_idx = [self.rdf_indexes[rdf] for rdf in pred_turn if len(rdf) == 3]
                ref_turn_idx = [self.rdf_indexes[rdf] for rdf in ref_turn if len(rdf) == 3]  # some slot values are not flawless rdfs in the annotated set
                missing_els_from_ref = set(pred_turn_idx) - set(ref_turn_idx)
                score = 0 if missing_els_from_ref else 1
                scores.append(score)
            else:
                scores.append(0)
        jga_turn_score = sum(scores) / len(scores) if len(scores) != 0 else 0
        self.turn_jga_scores.append(jga_turn_score)
        return self.joint_goal_accuracy(predictions[1:], references[1:])

    def index_encoding(self):
        """
        encodes all rdfs from preds and labels to vectorize evaluation 
        """
        pred_rdfs = [rdfs for batch in self.decoded_preds for rdfs in batch]
        pred_rdfs = set().union(*pred_rdfs)
        label_rdfs = [rdfs for batch in self.decoded_labels for rdfs in batch]
        label_rdfs = set().union(*label_rdfs)
        unique_rdfs = list(label_rdfs | pred_rdfs)
        invalid_val = 0
        self.rdf_indexes = dict()
        for i in range(len(unique_rdfs)):
            if len(unique_rdfs[i]) != 3:
                invalid_val -= 1
                self.rdf_indexes.setdefault(unique_rdfs[i], -i)
            else:
                self.rdf_indexes.setdefault(unique_rdfs[i], i)
        self.rdf_indexes

    def dialogue_reconstruction(self):
    
        dialogues = dict()
        if isinstance(self.dialogue_ids, int):
            if dialogue_id in dialogues:
                dialogues[self.dialogue_ids]["preds"].append(self.decoded_preds)
                dialogues[self.dialogue_ids]["labels"].append(self.decoded_labels)
            else:
                dialogues[self.dialogue_ids] = {"preds": [self.decoded_preds], "labels": [self.decoded_labels]}
        else:
           # flattening to avoid a nested loops
            dialogue_id = [d for batch in self.dialogue_ids for d in batch]
            pred = [p for batch in self.decoded_preds for p in batch]
            label = [l for batch in self.decoded_labels for l in batch]
            for diag_id, pr, lb in zip(dialogue_id, pred, label):
                if diag_id in dialogues:
                    dialogues[diag_id]["preds"].append(pr)
                    dialogues[diag_id]["labels"].append(lb)
                else:
                    dialogues[diag_id] = {"preds": [pr], "labels": [lb]}
    
        new_batch_preds = [dialogues[k]["preds"] for k in dialogues.keys()]  # v["preds"] if iterating over values instead?
        new_batch_labels = [dialogues[k]["labels"] for k in dialogues.keys()]
    
        self.ordered_dialogues = {"ordered_preds": new_batch_preds, "ordered_labels": new_batch_labels, "dialogue_id": dialogues.keys()}


    def store_model_predictions(self):
        states_df = pd.DataFrame(self.ordered_dialogues)
        states_df.to_csv("nested_states.csv", index=False)
