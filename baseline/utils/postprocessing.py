# Copyright 2023 Orange
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
