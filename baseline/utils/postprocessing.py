
def clean_slot_val(node):

    node = node.replace('_', '')
    node = node.replace(',', '')  # removing commas to facilitate state creation and splitting
    node = node.replace('=', '')  # removing commas to facilitate state creation and splitting
    return node.lower().strip()
    # masking random values given after search, result, etc.
    #mask = "****"  # Replace with your desired mask

def clean_state(state):
    # using = for eval purposes
    return '='.join([clean_slot_val(node) for node in state.split('=')])


def postprocess_states(decoded_batch):
    """
    returns several states per batch
    """
    decoded_batch = [clean_state(state) for row in decoded_batch for state in row.split('|')]
    return list(frozenset(decoded_batch))

    #raise SystemExit
    #decoded_batch = [row.split('|') for row in decoded_batch]  # why is there only one row?
    #decoded_batch = [frozenset(clean_state(states) for states in row) for row in decoded_batch]
    
    #return decoded_batch[0] if len(decoded_batch) == 1 else decoded_batch

