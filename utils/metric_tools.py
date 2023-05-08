import pandas as pd

class DSTMetrics:

    def __init__(self, outputs):
        self.decoded_labels = [out['labels'] for out in outputs]
        self.decoded_preds = [out['preds'] for out in outputs]
        self.dialogue_ids = [out['ids'] for out in outputs]
        self.slots_empty_assignment = ["none", '', ' ', '*']


    def __call__(self, store=False):
        self.index_encoding()
        self.dialogue_reconstruction()
        data = self.flatten_batches()

        self.slot_counts = {slot: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for slot in slot_names}
        no_context_compute = self.compute(data["preds"], data["labels"])
        context_compute = self.compute(data["ordered_preds"], data["ordered_labels"])

        if store:
            self.store_model_predictions()

        data.clear()

    def flatten_batches(self):
        flatten_preds = [rdf for pred in self.decoded_preds for rdf in pred]
        flatten_labels = [rdf for label in self.decoded_labels for rdf in label]

        ordered_preds = [rdf for pred in self.ordered_dialogues["ordered_preds"] for rdf in pred]
        ordered_labels = [rdf for label in self.ordered_dialogues["ordered_labels"] for rdf in label]

        return {"preds": flatten_preds, "labels": flatten_labels,
                "ordered_preds": ordered_preds, "ordered_labels": ordered_labels}

    def compute(self, preds, labels):

        self.all_jga_scores = []
        self.name_forgotten_measures = []
        self.name_invented_measures = []
        self.active_references = []
        self.active_predictions = []

        self.true_positives = []
        self.true_negatives = []
        self.false_positives = []
        self.false_negatives = []
        for pred_turn, ref_turn in zip(preds, labels):
            pred_turn_idx = [self.rdf_indexes[rdf] for rdf in pred_turn if len(rdf) == 3]
            ref_turn_idx = [self.rdf_indexes[rdf] for rdf in ref_turn if len(rdf) == 3]  # some slot values are not flawless rdfs in the annotated set
            #invented
            missing_els_from_ref = set(pred_turn_idx) - set(ref_turn_idx)
            #forgotten
            missing_els_from_pred = set(ref_turn_idx) - set(pred_turn_idx)

            # JGA
            self.joint_goal_accuracy(ref_turn, pred_turn, missing_els_from_ref)

            #slot_name_scores
            self.slot_name_scores(pred_turn_idx, ref_turn_idx, missing_els_from_pred, missing_els_from_ref)

            if pred_turn_idx and ref_turn_idx:
                pred_slots_values = [self.inv_index[idx][1:] for idx in pred_turn_idx]
                ref_slots_values = [self.inv_index[idx][1:] for idx in ref_turn_idx]

                # average_goal_accuracy
                self.average_goal_accuracy(pred_slot_values, ref_slot_values)

                # compute slot scores
                self.build_confusion_table(pred_slot_values, ref_slot_values)

        else:
            mean_jga = sum(self.all_jga_scores) / len(self.all_jga_scores) if len(self.all_jga_scores) != 0 else 0
            forgotten_mean_proportion = sum(self.name_forgotten_measures)/len(self.name_forgotten_measures)
            invented_mean_proportion = sum(self.name_invented_measures)/len(self.name_invented_measures)
            average_goal_accuracy = sum(i == j for i, j in zip(self.active_predictions, self.active_references)) / len(self.active_predictions) if self.active_predictions else 0

            # compute slot scores
            self.compute_slot_scores

        results = {"jga": mean_jga, "forgotten_mean_proportion": forgotten_mean_proportion,
                "invented_mean_proportion": invented_mean_proportion, "average_goal_accuracy": average_goal_accuracy}

        return results


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
        self.inv_index = {v: k for k, v in self.rdf_indexes.items()}

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


    def joint_goal_accuracy(self, ref_turn, pred_turn, missing_els_from_ref):
        if len(ref_turn) == len(pred_turn):
            jga_score = 0 if missing_els_from_ref else 1
            self.all_jga_scores.append(jga_score)
        else:
            jga_score = 0
            self.all_jga_scores.append(0)

    def slot_name_scores(self, pred_turn_idx, ref_turn_idx, missing_els_from_pred, missing_els_from_ref):
        turn_measures = dict()
        # forgotten
        if pred_turn_idx:
            turn_measures["forgotten"] = len(missing_els_from_pred) / len(pred_turn_idx)
            self.name_forgotten_measures.append(turn_measures["forgotten"])
        # invented
        if ref_turn_idx:
            turn_measures["invented"] = len(missing_els_from_ref) / len(ref_turn_idx)
            self.name_invented_measures.append(turn_measures["invented"])

    def average_goal_accuracy(self, pred_slot_values, ref_slot_values):

        #active_ref = {slot: value for slot, value in ref_slots_values if value not in self.slots_empty_assignment}
        active_ref = [(slot, value) for slot, value in ref_slots_values if value not in self.slots_empty_assignment]
        #active_pred = {slot: value for slot, value in pred_slots_values if slot in active_ref}
        active_pred = [(slot, value) for slot, value in pred_slots_values if slot in active_ref]
        if len(active_ref) != 0:
            self.active_references.append(active_ref)
            self.active_predictions.append(active_pred)
    
    def build_confusion_table(self, pred_slot_values, ref_slot_values):

            pred_slots, pred_values = list(zip(*pred_slot_values))
            ref_slots, ref_values = list(zip(*ref_slot_values))
            self.true_positives += [slot for i, slot in enumerate(pred_slots) if (slot in ref_slots) and (pred_values[i] == ref_values[i]) and (ref_values[i] not in self.slots_empty_assignment)]
            self.true_negatives += [slot for i, slot in enumerate(pred_slots) if (slot in ref_slots) and (pred_values[i] == ref_values[i]) and (ref_values[i] in self.slots_empty_assignment)]
            self.false_positives += [slot for i, slot in enumerate(pred_slots) if (slot not in ref_slots) or (slot in ref_slots and pred_values[i] != ref_values[i] and pred_values[slot] not in self.slots_empty_assignment)]
            self.false_negatives += [slot for i, slot in enumerate(ref_slots) if (slot not in pred_slots) and (ref_values[i] not in self.slots_empty_assignment)]

    def compute_slot_scores(self):
        pass

    def store_model_predictions(self):
        states_df = pd.DataFrame(self.ordered_dialogues)
        states_df.to_csv("nested_states.csv", index=False)




