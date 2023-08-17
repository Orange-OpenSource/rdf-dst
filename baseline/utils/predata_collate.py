from dataclasses import dataclass
from utils.postprocessing import clean_slot_val
import random


@dataclass
class BaselinePreDataCollator:
    
    def __init__(self, tokenizer, source_len, target_len, exp_setup, sys_response):

        self.sys_response = sys_response
        self.exp_setup = exp_setup
        self.source_len = source_len
        self.target_len = target_len


        self.slot_tkn = '<slot_tkn>'
        self.value_tkn = '<value_tkn>'

        sentinel_tkns = {"additional_special_tokens": [self.slot_tkn, self.value_tkn]}
        #tokenizer.add_special_tokens(sentinel_tkns)

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
        model_input = []
        turn_ids = []
        context = ''
        for i, t in enumerate(dialogue_data):
            user_slot_vals = [s_v for slot_val in t['user']['dialog-acts'] for s_v in slot_val['slots']] 
            sys_slot_vals = [s_v for slot_val in t['system']['dialog-acts'] for s_v in slot_val['slots']] 
            # Leo replaces old slots when they have a new value. This makes sense.
            # set way
            #slot_values = list(frozenset(clean_slot_val(s_v['name']) + '=' + clean_slot_val(s_v['value']) for s_v in user_slot_vals + sys_slot_vals))
            # leo's way...
            slot_values = {clean_slot_val(s_v['name']): clean_slot_val(s_v['value']) for s_v in user_slot_vals + sys_slot_vals}
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
                txt_input = context + curr_user if not self.sys_response else context + curr_user + curr_system
            elif self.exp_setup == 3:
                txt_input = prev_system + curr_user if not self.sys_response else prev_system + curr_user + curr_system
            elif self.exp_setup in [4, 5]:
                txt_input = curr_user if not self.sys_response else curr_user + curr_system

            model_input.append(txt_input.strip().lower())
        if self.exp_setup in [1, 3, 4]:
            first_turn = model_input[0]
            #txt_input = [txt + states[i] for i, txt in enumerate(txt_input[1:])]
            model_input = [txt + ' STATE ' + states[i] for i, txt in enumerate(model_input[1:])]
            model_input.insert(0, first_turn)
        elif self.exp_setup == 6:
            model_input = ['STATE ' + state for state in states[1:]]
            model_input.insert(0, ' ')

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
