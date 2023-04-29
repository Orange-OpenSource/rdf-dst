
def postprocess_rdfs(decoded_batch):

    decoded_batch = [row.split(',') for row in decoded_batch]
    decoded_batch = [[rdfs[i:i+3] for i in range(0, len(rdfs), 3)] for rdfs in decoded_batch]

    return decoded_batch

def joint_goal_accuracy(predictions, references):
    for pred, ref in zip(predictions, references):
        print(pred)
        print()
    
    return 69