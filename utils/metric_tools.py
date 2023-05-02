import re

def postprocess_rdfs(decoded_batch):
    """
    returns several rdf triplets per batch
    """

    regexSplit = re.compile(r"(?<!\s),(?!\s)")
    decoded_batch = [regexSplit.split(row) for row in decoded_batch]
    decoded_batch = [[word.strip() for word in rdfs] for rdfs in decoded_batch]
    return [[rdfs[i:i+3] for i in range(0, len(rdfs), 3)] for rdfs in decoded_batch]


class DSTMetrics:
    def joint_goal_accuracy(self, predictions, references):
        for preds, refs in zip(predictions, references):
            print(len(preds))
            print(len(refs))
            print(refs)
            break
        raise SystemExit
        return 69

# FOR NON-LINEAR RDF, WHAT TOD DO?
def nonlinear_jga(predictions, references):
    avg_goal_pred_score = 0
    for preds, refs in zip(predictions, references):
        for pred_rdfs, ref_rdfs in zip(preds, refs):
            print("PREDICTIONS")
            print(pred_rdfs)
            print(len(ref_rdfs))
            print("REFERENCES")
            print(ref_rdfs)
            print(len(pred_rdfs))
            for pr_rdf, ref_rdf in zip(pred_rdfs, ref_rdfs):
                if (ref_rdf in pred_rdfs) and pr_rdf:  # checking that pr_rdf has several elements
                    print("CORRECT")
                    avg_goal_pred_score += 1

                    # reducing size of arrays to facilitate complexity?
                    ref_rdfs = [rdf for rdf in ref_rdfs if rdf != ref_rdf]
                    pred_rdfs = [rdf for rdf in pred_rdfs if rdf != ref_rdf]
                    print(len(ref_rdfs))
                    print(len(pred_rdfs))
                break
            break
    
    return 69