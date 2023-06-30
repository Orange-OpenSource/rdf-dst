from collections import OrderedDict

def clean_slot_val(node):

    node = node.replace('_', '')
    node = node.replace(',', '')  # removing commas to facilitate state creation and splitting
    node = node.replace('=', '')  # removing commas to facilitate state creation and splitting
    return node.lower().strip()

def clean_state(state):
    # using = for eval purposes
    if '=' in state:
        return [clean_slot_val(node) for node in state.split('=')]
    else:
        # cleaning bad pred anyway. Replaced with None, no need for this
        #return ['_NONE_', clean_slot_val(state)]
        return None


def postprocess_states(decoded):
    """
    returns several states per batch
    """
    if '|' in decoded:
        clean_states = [clean_state(state) for state in decoded.split('|')]
        #return {slot_val[0]: slot_val[1] for slot_val in (clean_state(state) for state in decoded.split('|'))}

    else:
        clean_states = clean_state(decoded)
    if clean_states:
    # swap for OrderedDict?  there are repeated keys but most recent state is what matters. See https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/44018.pdf
    #clean_states = {slot_val[0]: slot_val[1] for slot_val in clean_states if slot_val}
        clean_states = OrderedDict((slot_val[0], slot_val[1]) for slot_val in clean_states if slot_val)

        return frozenset(slot + '=' + val for slot, val in clean_states.items())  # we turn them into sets because it is easier to evaluate
    # super strict match with old states, but unneeded atm...
    #return set([s[0]+ '=' + s[1] for s in clean_states])
    else:
        # returning fake _NONE_ to enable evaluation...
        return {'_NONE_'}
