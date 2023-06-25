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
        self.slot_tkn = '<slot_tkn>'
        self.val_tkn = '<val_tkn>'

        sentinel_tkns = {"additional_special_tokens": [self.user_tkn, self.sys_tkn, self.slot_tkn, self.val_tkn]}

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

            for txt, s_v, turn_id in zip(txt_input, slot_value, turn_ids):
                tokenized = self.tokenize(txt, s_v)

                input_ids.append(tokenized['input_ids'])
                attention_mask.append(tokenized['attention_mask'])
                labels.append(tokenized['labels'])

                dialogue_ids.append(diag_id)
                turn_number.append(turn_id)


        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': labels, 'dialogue_id': dialogue_ids, 'turn_number': turn_number}

    
    def create_inputs_outputs(self, dialogue_data):

        states = []
        txt_input = []
        turn_ids = []
        context = ''
        for i in dialogue_data['turn_id']:
            idx = int(i)
            turn_ids.append(idx)
            state = dialogue_data['belief_state'][idx]
            seq_states = []
            for slot, val in zip(state['slot'], state['value']):
                slot = clean_slot_val(slot)
                val = clean_slot_val(val)
                seq_states.append(self.slot_tkn + slot + '=' + self.val_tkn + val)

            # augmentation
            seq_states = random.sample(seq_states, len(seq_states))
            
            states.append('|'.join(seq_states))

            if self.exp_setup in [1, 2]:
                system = dialogue_data['sys_utterance'][idx]
                user = dialogue_data['usr_utterance'][idx]
                convo = self.sys_tkn + system + self.user_tkn + user
                context += convo
                txt_input.append(context)
        
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
