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

from dataclasses import dataclass
from utils.postprocessing import clean_slot_val
import random


@dataclass
class BaselinePreDataCollator:
    
    def __init__(self, tokenizer, source_len, target_len, exp_setup):

        self.exp_setup = exp_setup
        self.source_len = source_len
        self.target_len = target_len
        self.user_tkn = '<user_tkn>'
        self.sys_tkn = '<sys_tkn>'
        #self.slot_tkn = '<slot_tkn>'
        #self.val_tkn = '<val_tkn>'

        #sentinel_tkns = {"additional_special_tokens": [self.user_tkn, self.sys_tkn, self.slot_tkn, self.val_tkn]}
        sentinel_tkns = {"additional_special_tokens": [self.user_tkn, self.sys_tkn]}

        tokenizer.add_special_tokens(sentinel_tkns)
        self.tokenizer = tokenizer

    def __call__(self, batch):
        
        input_ids = []
        attention_mask = []
        labels = []
        dialogue_ids = []
        turn_number = []


        for diag_id, dialogue_data in zip(batch['dialogue_id'], batch['turns']):  # dict, history is a str that is the key
            txt_input, slot_value, turn_ids = self.create_inputs_outputs(dialogue_data)

            turn_number.extend(turn_ids)
            dialogue_ids.extend([diag_id] * len(turn_ids))
            for txt, s_v in zip(txt_input, slot_value):
                tokenized = self.tokenize(txt, s_v)

                input_ids.append(tokenized['input_ids'])
                attention_mask.append(tokenized['attention_mask'])
                labels.append(tokenized['labels'])


        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': labels, 'dialogue_id': dialogue_ids, 'turn_number': turn_number}

    
    def create_inputs_outputs(self, dialogue_data):

        states = []
        txt_input = []
        turn_ids = []
        context = ''
        for t in dialogue_data:
            user_slot_vals = [s_v for slot_val in t['user']['dialog-acts'] for s_v in slot_val['slots']] 
            sys_slot_vals = [s_v for slot_val in t['system']['dialog-acts'] for s_v in slot_val['slots']] 
            # Leo replaces old slots when they have a new value. This makes sense.
            #slot_values = list(frozenset(clean_slot_val(s_v['name']) + '=' + clean_slot_val(s_v['value']) for s_v in user_slot_vals + sys_slot_vals))
            slot_values = {clean_slot_val(s_v['name']): clean_slot_val(s_v['value']) for s_v in user_slot_vals + sys_slot_vals}
            slot_values = [slot + '=' + value for slot, value in slot_values.items()]
            # augmentation: does it make eval more complicated?
            #slot_values = random.sample(slot_values, len(slot_values))
            states.append(' ; '.join(slot_values))
            turn_ids.append(t['turn-index'])

            if self.exp_setup in [1, 2]:
                system = t['system']['text']
                user = t['user']['text']
                convo = self.sys_tkn + system + self.user_tkn + user
                context += convo
                txt_input.append(context.strip().lower())
        if self.exp_setup == 1:
            first_turn = txt_input[0]
            txt_input = [txt + states[i] for i, txt in enumerate(txt_input[1:])]
            txt_input.insert(0, first_turn)
        elif self.exp_setup == 3:
            txt_input = [' ']
            txt_input.extend(states[:-1])
            
        return txt_input, states, turn_ids


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
