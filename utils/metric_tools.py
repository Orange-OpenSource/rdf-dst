import re

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
    def joint_goal_accuracy(self, predictions, references):
        print(len(predictions))
        print(len(references))
        print(references[3][-1])
        print("TITO")
        print(predictions[3][-1])
        raise SystemExit
        print()
        for preds, refs in zip(predictions, references):
            for pred_rdfs, ref_rdfs in zip(preds[::-1], refs[::-1]):
                print(pred_rdfs)
                print("LABELS")
                print(ref_rdfs)
                for rdf in pred_rdfs:
                    if rdf in ref_rdfs:
                        print("Correct")
                        print(rdf)
                raise SystemExit
        print("CONTINUE WITH COMPLEX REP TO COMPARE")
        #raise SystemExit
        #for preds, refs in zip(predictions, references):
        #    print(len(preds))
        #    print(len(refs))
        #    #print(refs)
        #    break
        #raise SystemExit
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