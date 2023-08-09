from dataclasses import dataclass
from utils.postprocessing import clean_slot_val
import random


@dataclass
class BaselinePreDataCollator:
    
    def __init__(self, tokenizer, source_len, target_len, exp_setup, inference_time=False):

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
        txt_input = []
        turn_ids = []
        context = ''
        for t in dialogue_data:
            user_slot_vals = [s_v for slot_val in t['user']['dialog-acts'] for s_v in slot_val['slots']] 
            sys_slot_vals = [s_v for slot_val in t['system']['dialog-acts'] for s_v in slot_val['slots']] 
            # Leo replaces old slots when they have a new value. This makes sense.
            # set way
            slot_values = list(frozenset(clean_slot_val(s_v['name']) + '=' + clean_slot_val(s_v['value']) for s_v in user_slot_vals + sys_slot_vals))
            # leo's way...
            #slot_values = {clean_slot_val(s_v['name']): clean_slot_val(s_v['value']) for s_v in user_slot_vals + sys_slot_vals}
            #slot_values = [f'{slot}={value}' for slot, value in slot_values.items()]

            # augmentation: does it make eval more complicated?
            slot_values = random.sample(slot_values, len(slot_values))
            states.append(';'.join(slot_values))
            turn_ids.append(t['turn-index'])

            system = t['system']['text']
            user = t['user']['text']
            convo = 'SYSTEM: ' + system + 'USER: ' + user
            context += convo
            if self.exp_setup in [1, 2]:
                txt_input.append(context.strip().lower())
            elif self.exp_setup == 3:
                txt_input.append(convo.strip().lower())
        if self.exp_setup in [1, 3]:
            first_turn = txt_input[0]
            #txt_input = [txt + states[i] for i, txt in enumerate(txt_input[1:])]
            txt_input = [txt + 'STATE: ' + states[i] for i, txt in enumerate(txt_input[1:])]
            txt_input.insert(0, first_turn)
        
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
