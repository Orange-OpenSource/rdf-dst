# Copyright (c) 2023 Orange

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITEDTOTHE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Software Name : knowledge-graph-dst
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: H. Andres Gonzalez


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

    # leo's way
    post_states = dict()
    # set way
    post_set_states = set()

    for slot_val in clean_states:
        if len(slot_val) == 2:
            post_states[slot_val[0]] = slot_val[1]
            new_slot_val = f'{slot_val[0]}={slot_val[1]}'
            post_set_states.add(new_slot_val)
        else:
            post_states['_NONE_'] = '_NONE_'
            post_set_states.add('_NONE_')
    
    return post_states
    #return post_set_states
