import re
import pandas as pd

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

def store_model_predictions(all_dialogue_states):
    states_df = pd.DataFrame(all_dialogue_states)
    states_df.to_csv("nested_states.csv", index=False)

def dialogue_reconstruction(dialogue_id, pred, label):

    dialogues = dict()
    if isinstance(dialogue_id, int):
        if dialogue_id in dialogues:
            dialogues[dialogue_id]["preds"].append(pred)
            dialogues[dialogue_id]["labels"].append(label)
        else:
            dialogues[diag_id] = {"preds": [pred], "labels": [label]}
    else:
       # flattening to avoid a nested loops
        dialogue_id = [d for batch in dialogue_id for d in batch]
        pred = [p for batch in pred for p in batch]
        label = [l for batch in label for l in batch]
        for diag_id, pr, lb in zip(dialogue_id, pred, label):
            if diag_id in dialogues:
                dialogues[diag_id]["preds"].append(pr)
                dialogues[diag_id]["labels"].append(lb)
            else:
                dialogues[diag_id] = {"preds": [pr], "labels": [lb]}

    
    new_batch_preds = [dialogues[k]["preds"] for k in dialogues.keys()]  # v["preds"] if iterating over values instead?
    new_batch_labels = [dialogues[k]["labels"] for k in dialogues.keys()]

    return {"ordered_preds": new_batch_preds, "ordered_labels": new_batch_labels, "dialogue_id": dialogues.keys()}

