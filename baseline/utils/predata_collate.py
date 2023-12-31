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


from dataclasses import dataclass
from utils.postprocessing import clean_slot_val
import random
import re


@dataclass
class BaselinePreDataCollator:
    
    def __init__(self, tokenizer, source_len, target_len, exp_setup, dataset_type, include_prev_sys=True):

        self.exp_setup = exp_setup
        self.include_prev_sys = include_prev_sys
        data_collation = {"multiwoz": self.multiwoz_loop, "dstc2": self.dstc2_loop, "sfxdial": self.sfx_loop}
        self.data_collation = data_collation[dataset_type]
        self.source_len = source_len
        self.target_len = target_len


        self.slot_tkn = '<slot_tkn>'
        self.value_tkn = '<value_tkn>'

        sentinel_tkns = {"additional_special_tokens": [self.slot_tkn, self.value_tkn]}
        #tokenizer.add_special_tokens(sentinel_tkns)

        self.tokenizer = tokenizer

    def __call__(self, batch):
        
        return self.data_collation(batch)
    
    def dstc2_loop(self, batch):
        # system greets first
        input_ids = []
        attention_mask = []
        labels = []
        dialogue_ids = []
        turn_number = []

        for diag_id, dialogue_data in zip(batch['session-id'], batch['turns']):  # dict, history is a str that is the key
            txt_input, slot_value, turn_ids = self.create_inputs_outputs_dstc2(dialogue_data)

            turn_number.extend(turn_ids)
            dialogue_ids.extend([diag_id] * len(turn_ids))
            for txt, s_v in zip(txt_input, slot_value):
                tokenized = self.tokenize(txt, s_v)

                input_ids.append(tokenized['input_ids'])
                attention_mask.append(tokenized['attention_mask'])
                labels.append(tokenized['labels'])

        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': labels, 'dialogue_id': dialogue_ids, 'turn_number': turn_number}
    
    def model_input_processing(self, model_input, states):

        if self.exp_setup in [1, 4]:
            # non linearized, they are in a list so tokenizer can work with these
            first_turn = model_input[0]
            prev_states = ['STATE ' + state if state != '' else state for state in states[:-1]]

            model_input = [txt + ' ' + s for txt, s in zip(model_input[1:], prev_states)]
            model_input.insert(0, first_turn)
        
        return model_input

    def create_inputs_outputs_dstc2(self, dialogue_data):

        # sys slot vals before user input, so it's fine here because the system greets first
        states = []
        model_input = []
        turn_ids = []
        context = ''
        for i, t in enumerate(dialogue_data):
            system = t['output']
            user = t['label']

            sys_slot_vals = [s_v['slots'] for s_v in system['dialog-acts'] if s_v['slots']]
            user_slot_vals = [s_v['slots'] for s_v in user['dialog-acts'] if s_v['slots']]
            if sys_slot_vals:
                sys_slot_vals = sys_slot_vals[0] if isinstance(sys_slot_vals[0], list) and len(sys_slot_vals) == 1 else sys_slot_vals
                sys_slot_vals = {sv['name']: sv['value'] for sv in sys_slot_vals if isinstance(sv, dict)}
            if user_slot_vals:
                user_slot_vals = user_slot_vals[0] if isinstance(user_slot_vals[0], list) and len(user_slot_vals) == 1 else user_slot_vals
                user_slot_vals = {sv['name']: sv['value'] for sv in user_slot_vals if isinstance(sv, dict)}
                #except TypeError:
                #    print("TITO")
                #    print()
                #    print(user_slot_vals)
                #    print("JIJI")
                #    raise SystemExit


            if user_slot_vals and sys_slot_vals:
                missing_slots = set(sys_slot_vals.keys()) - set(user_slot_vals.keys())
                sys_slot_vals = {slot: sys_slot_vals[slot] for slot in missing_slots}
            
            elif not isinstance(user_slot_vals, dict):
                user_slot_vals = {}

            elif not isinstance(sys_slot_vals, dict):
                sys_slot_vals = {}

            slot_values = {**sys_slot_vals, **user_slot_vals}
            slot_values = [f'{clean_slot_val(slot)}={clean_slot_val(value)}' for slot, value in slot_values.items()]
            slot_values = random.sample(slot_values, len(slot_values))

            states.append(';'.join(slot_values))

            turn_ids.append(i)
            sys_utterance = 'SYSTEM ' + system['transcript'] + ' ' if system['transcript'] else ''
            user_utterance = 'USER ' + user['transcription'] + ' ' if user['transcription'] else ''

            default_input = sys_utterance + user_utterance

            context += default_input

            if self.exp_setup in [1, 2]:
                txt_input = context + default_input
            elif self.exp_setup in [4, 5]:
                txt_input = default_input

            model_input.append(txt_input.strip().lower())

        model_input = self.model_input_processing(model_input, states)

        return model_input, states, turn_ids

    def sfx_loop(self, batch):
        # system greets first
        input_ids = []
        attention_mask = []
        labels = []
        dialogue_ids = []
        turn_number = []

        for diag_id, dialogue_data in zip(batch['id'], batch['dial']):  # dict, history is a str that is the key
            txt_input, slot_value, turn_ids = self.create_inputs_outputs_sfx(dialogue_data)

            turn_number.extend(turn_ids)
            dialogue_ids.extend([diag_id] * len(turn_ids))
            for txt, s_v in zip(txt_input, slot_value):
                tokenized = self.tokenize(txt, s_v)

                input_ids.append(tokenized['input_ids'])
                attention_mask.append(tokenized['attention_mask'])
                labels.append(tokenized['labels'])

        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': labels, 'dialogue_id': dialogue_ids, 'turn_number': turn_number}
    
    def helper_sfx_func(self, raw_slot_values):
        pattern = re.compile(r'\((.*?)\)')
        slot_vals = []
        for raw_sys in raw_slot_values:
            mo = pattern.search(raw_sys)
            if mo:
                new_sv = mo.group(1)

                new_sv = new_sv.split(';')
                new_sv = {s_v.split('=')[0] if '=' in s_v else s_v: s_v.split('=')[1] if '=' in s_v else '?' for s_v in new_sv} 
                new_sv = {s.strip("'") if s.startswith("'") and s.endswith("'") else s: v.strip("'") if v.startswith("'") and v.endswith("'") else v for s, v in new_sv.items()}
                slot_vals.append(new_sv)
        return {slot: value for dictionary in slot_vals for slot, value in dictionary.items() if slot}

    def create_inputs_outputs_sfx(self, dialogue_data):

        # sys slot vals before user input, so it's fine here because the system greets first
        states = []
        model_input = []
        turn_ids = []
        context = ''

        for i, d in enumerate(dialogue_data):
            raw_sys_vals = d['S']['dact']
            raw_user_vals = d['U']['dact']
            sys_slot_vals = self.helper_sfx_func(raw_sys_vals)
            user_slot_vals = self.helper_sfx_func(raw_user_vals)

            missing_slots = set(sys_slot_vals.keys()) - set(user_slot_vals.keys())
            sys_slot_vals = {slot: sys_slot_vals[slot] for slot in missing_slots}

            slot_values = {**sys_slot_vals, **user_slot_vals}
            slot_values = [f'{clean_slot_val(slot)}={clean_slot_val(value)}' for slot, value in slot_values.items()]
            slot_values = random.sample(slot_values, len(slot_values))

            states.append(';'.join(slot_values))
            turn_ids.append(i)

            raw_sys_utter = d['S']['base']
            raw_user_utter = d['U']['hyp']
            sys_utterance = 'SYSTEM ' + raw_sys_utter + ' ' if raw_sys_utter else ''
            user_utterance = 'USER ' +  raw_user_utter + ' ' if raw_user_utter else ''

            default_input = sys_utterance + user_utterance

            context += default_input

            if self.exp_setup in [1, 2]:
                txt_input = context + default_input
            elif self.exp_setup in [4, 5]:
                txt_input = default_input

            model_input.append(txt_input.strip().lower())

        model_input = self.model_input_processing(model_input, states)


        return model_input, states, turn_ids

    def multiwoz_loop(self, batch):
        # user greets first
        input_ids = []
        attention_mask = []
        labels = []
        dialogue_ids = []
        turn_number = []


        for diag_id, dialogue_data in zip(batch['dialogue_id'], batch['turns']):  # dict, history is a str that is the key
            txt_input, slot_value, turn_ids = self.create_inputs_outputs_multiwoz(dialogue_data)

            turn_number.extend(turn_ids)
            dialogue_ids.extend([diag_id] * len(turn_ids))
            for txt, s_v in zip(txt_input, slot_value):
                tokenized = self.tokenize(txt, s_v)

                input_ids.append(tokenized['input_ids'])
                attention_mask.append(tokenized['attention_mask'])
                labels.append(tokenized['labels'])

        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': labels, 'dialogue_id': dialogue_ids, 'turn_number': turn_number}


    def create_inputs_outputs_multiwoz(self, dialogue_data):

        states = []
        model_input = []
        turn_ids = []
        context = ''
        prev_sys_slot_val = []
        for i, t in enumerate(dialogue_data):
            user_slot_vals = [s_v for slot_val in t['user']['dialog-acts'] for s_v in slot_val['slots']] 
            curr_slot_vals = user_slot_vals + prev_sys_slot_val

            # sys slot values include updates to user slot values based on system responses as conv goes on.
            # WARNING: current sys_sv is made up of the curr system response, thus we must use the previous ones
            if self.include_prev_sys:
                sys_slot_vals = [s_v for slot_val in t['system']['dialog-acts'] for s_v in slot_val['slots']] 
                prev_sys_slot_val = sys_slot_vals  # updating previous slot vals for next turn!
            # Leo replaces old slots when they have a new value. This makes sense.
            # set way
            #slot_values = list(frozenset(clean_slot_val(s_v['name']) + '=' + clean_slot_val(s_v['value']) for s_v in user_slot_vals))
            # leo's way...
            slot_values = {clean_slot_val(s_v['name']): clean_slot_val(s_v['value']) for s_v in curr_slot_vals}
            slot_values = [f'{slot}={value}' for slot, value in slot_values.items()]

            # augmentation: does it make eval more complicated?
            slot_values = random.sample(slot_values, len(slot_values))

            states.append(';'.join(slot_values))
            turn_ids.append(t['turn-index'])

            # UNUSED
            curr_system = 'SYSTEM ' + t['system']['text'] + ' '

            curr_user = 'USER ' + t['user']['text'] + ' '
            prev_system = ''
            if i > 0:
                prev_system = dialogue_data[i-1]['system']['text']
                prev_user = dialogue_data[i-1]['user']['text']
                prev_system = 'SYSTEM ' + prev_system + ' '
                prev_user = 'USER ' + prev_user + ' '
                convo = prev_user + prev_system
                context += convo

            if self.exp_setup in [1, 2]:
                txt_input = context + curr_user
            elif self.exp_setup == 3:
                txt_input = prev_system + curr_user
            elif self.exp_setup in [4, 5]:
                txt_input = curr_user

            model_input.append(txt_input.strip().lower())

        model_input = self.model_input_processing(model_input, states)
        

        return model_input, states, turn_ids


    def tokenize(self, dialogue : str, slot_value : str):
        
        encoding = self.tokenizer(dialogue,
                       is_split_into_words=False,
                       padding='max_length',
                       truncation=True,
                       max_length=self.source_len)
        
        target_encoding = self.tokenizer(slot_value, padding='max_length',
                                         is_split_into_words=False,
                                         truncation=True,
                                         max_length=self.target_len)
        
        labels = target_encoding.input_ids
        labels = [-100 if label == self.tokenizer.pad_token_id else label for label in labels]

        encoding['labels'] = labels

        return encoding
