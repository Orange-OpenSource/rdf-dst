import pandas as pd

class DSTMetrics:

    def __init__(self, outputs):
        self.decoded_labels = [out['labels'] for out in outputs]
        self.decoded_preds = [out['preds'] for out in outputs]
        self.dialogue_ids = [out['ids'] for out in outputs]
        self.slots_empty_assignment = ["none", '', ' ', '*']

    def __call__(self):
        self.index_encoding()
        self.dialogue_reconstruction()


    def flatten_batches(self):
        flatten_preds = [rdf for pred in self.decoded_preds for rdf in pred]
        flatten_labels = [rdf for label in self.decoded_labels for rdf in label]

        ordered__preds = [rdf for pred in self.ordered_dialogues["ordered_preds"] for rdf in pred]
        ordered_labels = [rdf for label in self.ordered_dialogues["ordered_labels"] for rdf in label]

    def compute(self, store=False):


        self.turn_jga_scores = []
        self.forg_mean_prop_turn = []
        self.inv_mean_prop_turn = []
        self.turn_average_goal = []

        no_context_results = self.metrics(self.decoded_preds, self.decoded_labels)
        no_context_jga = round(no_context_results["jga"] * 100, 3)
        no_context_forgotten_mean = round(no_context_results["forgotten_mean_proportion"] * 100, 3)
        no_context_invented_mean = round(no_context_results["invented_mean_proportion"] * 100, 3)
        # reinit the list for recursion eval with contextual batches
        self.turn_jga_scores = []
        self.active_references = []
        self.active_predictions = []

        context_results = self.metrics(self.ordered_dialogues["ordered_preds"], self.ordered_dialogues["ordered_labels"])
        context_jga = round(context_results["jga"] * 100, 3)
        context_forgotten_mean = round(context_results["forgotten_mean_proportion"] * 100, 3)
        context_invented_mean = round(context_results["invented_mean_proportion"] * 100, 3)
        self.decoded_preds.clear()
        self.decoded_labels.clear()
        self.dialogue_ids.clear()
        if store:
            self.store_model_predictions()
        self.ordered_dialogues.clear()
        return {"contextual_jga": context_jga, "no_contextual_jga": no_context_jga,
                "context_forgotten_mean": context_forgotten_mean, "context_invented_mean": context_invented_mean,
                "no_context_forgotten_mean": no_context_forgotten_mean, "no_context_invented_mean": no_context_invented_mean}

    def metrics(self, predictions, references):
        """

        the missing elements are used to compute the slot_name_scores as well
        """
        if (not predictions) and (not references):
            epoch_jga = sum(self.turn_jga_scores) / len(self.turn_jga_scores)
            epoch_forg_mean = sum(self.forg_mean_prop_turn) / len(self.forg_mean_prop_turn)
            epoch_inv_mean = sum(self.inv_mean_prop_turn) / len(self.inv_mean_prop_turn)
            # all predictions in references, have to rework logic...
            epoch_average_goal_accuracy = sum(self.turn_average-goal) / len(self.turn_average_goal)
            results = {"jga": epoch_jga, "forgotten_mean_proportion": epoch_forg_mean,
                    "invented_mean_proportion": epoch_inv_mean, "average_goal_accuracy": epoch_average_goal_accuracy}
            return results

        batch_preds = predictions[0]
        batch_refs = references[0]
        self.jga_scores = []
        self.name_forgotten_measures = []
        self.name_invented_measures = []
        self.active_references = []
        self.active_predictions = []
        for pred_turn, ref_turn in zip(batch_preds, batch_refs):
            pred_turn_idx = [self.rdf_indexes[rdf] for rdf in pred_turn if len(rdf) == 3]
            ref_turn_idx = [self.rdf_indexes[rdf] for rdf in ref_turn if len(rdf) == 3]  # some slot values are not flawless rdfs in the annotated set
            #invented
            self.missing_els_from_ref = set(pred_turn_idx) - set(ref_turn_idx)
            #forgotten
            self.missing_els_from_pred = set(ref_turn_idx) - set(pred_turn_idx)

            # JGA
            self.joint_goal_scores(ref_turn, pred_turn)

            # slot_name_scores
            self.slot_name_scores(pred_turn_idx, ref_turn_idx)

            # slot_name_scores
            # 1 because that's the slot, returning slots

            # average goal accuracy
            self.average_goal_accuracy(pred_turn_idx, ref_turn_idx)

        jga_turn_score = sum(self.jga_scores) / len(self.jga_scores) if len(self.jga_scores) != 0 else 0
        self.turn_jga_scores.append(jga_turn_score)

        forgotten_mean_proportion = sum(self.name_forgotten_measures)/len(self.name_forgotten_measures)
        invented_mean_proportion = sum(self.name_invented_measures)/len(self.name_invented_measures)
        self.forg_mean_prop_turn.append(forgotten_mean_proportion)
        self.inv_mean_prop_turn.append(invented_mean_proportion)


        average_goal_accuracy = sum(i == j for i, j in zip(self.active_predictions, self.active_references)) / len(self.active_predictions) if self.active_predictions else 0
        self.turn_average_goal.append(average_goal_accuracy)

        return self.metrics(predictions[1:], references[1:])

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


    def store_model_predictions(self):
        states_df = pd.DataFrame(self.ordered_dialogues)
        states_df.to_csv("nested_states.csv", index=False)


    def joint_goal_scores(self, reference_turn, prediction_turn):
        if len(reference_turn) == len(prediction_turn):
    
            jga_score = 0 if self.missing_els_from_ref else 1
            self.jga_scores.append(jga_score)
        else:
            self.jga_scores.append(0)

    def slot_name_scores(self, pred_turn_idx, ref_turn_idx):
        turn_measures = dict()
        # forgotten
        if pred_turn_idx:
            turn_measures["forgotten"] = len(missing_els_from_pred) / len(pred_turn_idx)
            self.name_forgotten_measures.append(turn_measures["forgotten"])
        # invented
        if ref_turn_idx:
            turn_measures["invented"] = len(missing_els_from_ref) / len(ref_turn_idx)
            self.name_invented_measures.append(turn_measures["invented"])

    def average_goal_accuracy(self, pred_turn_idx, ref_turn_idx):

        if pred_turn_idx and ref_turn_idx:
            pred_slots_values = [self.inv_index[idx][1:] for idx in pred_turn_idx]
            ref_slots_values = [self.inv_index[idx][1:] for idx in ref_turn_idx]
            #active_ref = {slot: value for slot, value in ref_slots_values if value not in self.slots_empty_assignment}
            active_ref = [(slot, value) for slot, value in ref_slots_values if value not in self.slots_empty_assignment]
            #active_pred = {slot: value for slot, value in pred_slots_values if slot in active_ref}
            active_pred = [(slot, value) for slot, value in pred_slots_values if slot in active_ref]
            if len(active_ref) != 0:
                self.active_references.append(active_ref)
                self.active_predictions.append(active_pred)
