from collections import OrderedDict

def clean_slot_val(node):

    node = node.strip()
    node = node.replace('_', '')
    node = node.replace(',', '')  # removing commas to facilitate state creation and splitting
    node = node.replace(';', '')  # removing commas to facilitate state creation and splitting
    node = node.replace('=', '')  # removing commas to facilitate state creation and splitting

    node = node.replace(':', '')
    node = node.replace('USER', '') 
    node = node.replace('SYSTEM', '') 
    node = node.replace('STATE', '') 
    return node.lower().strip()

def clean_state(state):
    # using = for eval purposes
    return [clean_slot_val(node) for node in state.split('=')]

def postprocess_states(decoded):
    """
    returns several states per batch
    """
    clean_states = [clean_state(state) for state in decoded.split(';')]
        #return {slot_val[0]: slot_val[1] for slot_val in (clean_state(state) for state in decoded.split('|'))}

    post_states = dict()
    for slot_val in clean_states:
        if len(slot_val) == 2:
            post_states[slot_val[0]] = slot_val[1]
        else:
            post_states['_NONE_'] = '_NONE_'
    
    return post_states
    # swap for OrderedDict?  there are repeated keys but most recent state is what matters. See https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/44018.pdf
    #clean_states = {slot_val[0]: slot_val[1] for slot_val in clean_states if slot_val and (len(slot_val) == 2)}
    #clean_states = OrderedDict((slot_val[0], slot_val[1]) for slot_val in clean_states if slot_val and (len(slot_val) == 2))

    #return frozenset(slot + '=' + val for slot, val in postprocessed_states.items())  # we turn them into sets because it is easier to evaluate
    #return {'_NONE_'}
    # super strict match with old states, but unneeded atm...
    #return set([s[0]+ '=' + s[1] for s in clean_states])
    # returning fake _NONE_ to enable evaluation...