def postprocess_rdfs(decoded_batch):

    decoded_batch = [row.split(',') for row in decoded_batch]
    decoded_batch = [[rdfs[i:i+3] for i in range(0, len(rdfs), 3)] for rdfs in decoded_batch]

    return decoded_batch

def joint_goal_accuracy(predictions, references):
    for preds, refs in zip(predictions, references):
        print("BATCH\n")
        print(len(preds))
        print("labels")
        print('\n'*2)
        print(len(refs))
        for pred_rdfs, ref_rdfs in zip(preds, refs):
            print(pred_rdfs)
            print(ref_rdfs)
            for pr_rdf, ref_rdf in zip(pred_rdfs, ref_rdfs):
                if ref_rdf in pred_rdfs:
                    print(ref_rdf)
                break
            break
    
    return 69

class DSTMetrics:

    def __init__(self):
        self.generated_states = dict()
        self.reference_states = dict()

    def add_batch(self, generated, references):
        self.generated_states["generated"] = generated
        self.reference_states["references"] = references