from dataclasses import dataclass
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class PreDataCollator:
    
    def __init__(self, tokenizer, max_len, history):

        self.max_len = max_len
        self.history = history
        self.user_tkn = '<user_tkn>'
        self.sys_tkn = '<sys_tkn>'
        self.state_tkn = '<state_tkn>'
        sentinel_tkns = {"additional_special_tokens": [self.user_tkn, self.sys_tkn, self.state_tkn]}
        tokenizer.add_special_tokens(sentinel_tkns)
        self.tokenizer = tokenizer
        logging.info(tokenizer.get_sentinel_tokens)

        
    
    def __call__(self, batch):
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for conversation in batch[self.history]:  # dict, history is a str that is the key
            conv_txt, txt_2_gen = self.create_inputs(conversation)
            tokenized = self.tokenize(conv_txt, txt_2_gen)
            input_ids.append(tokenized['input_ids'])
            attention_mask.append(tokenized['attention_mask'])
            labels.append(tokenized['labels'])

        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}


    def create_inputs(self, conversation):
        """
        This is where we choose the inputs and the rdf we will predict.
        Since we are using the whole state history and the first state cannot be predicted
        with a previous state, we initialize with an empty state and try to predict current state
        """

        turn_zero = conversation[0]
        sys_greet = self.sys_tkn + ' ' + turn_zero['S'] + ' ' + self.sys_tkn + ' '
        user_greet = self.user_tkn + ' ' + turn_zero['U'] + ' ' + self.user_tkn + ' '
        init_state = self.state_tkn + ' ' + self.state_tkn + '\n'
        first_rdf = turn_zero['rdf-state']
        # we can flatten all of the rdf-states and treat them as strings. But maybe the only last one matters?
        first_state = ', '.join(first_rdf[-1])
        txt_2_gen = self.state_tkn + ' ' + first_state + ' ' + self.state_tkn + '\n'
        conv_txt = sys_greet + user_greet + init_state
        for i in range(1, len(conversation)):
            turn = conversation[i]
            previous_state = conversation[i-1]['rdf-state']
            current_state = conversation[i]['rdf-state']
            system = turn['S']
            user = turn['U']
            prev_rdf = ', '.join(previous_state[-1])
            curr_rdf = ', '.join(current_state[-1])
            conv_txt += self.sys_tkn + ' ' +  system + ' ' + self.sys_tkn + ' '
            conv_txt += self.user_tkn + ' ' + user + ' ' + self.user_tkn + ' '
            conv_txt += self.state_tkn + ' ' + prev_rdf + ' ' + self.state_tkn + '\n'
            txt_2_gen += self.state_tkn + ' ' + curr_rdf + ' ' + self.state_tkn + '\n'

        return conv_txt, txt_2_gen

    def tokenize(self, conv, curr_states):
        
        input_txt = conv.strip().split()  
        output_txt = curr_states.split() 

        # using tokenizer to encode sentence (includes padding/truncation up to max length)
        encoding = self.tokenizer(input_txt,
                       text_target=output_txt,
                       is_split_into_words=True,
                       padding='max_length',
                       truncation=True,
                       max_length=self.max_len)


        # no need to align or ignore tokens with -100!
        #encoding['labels'] = [-100 if label == 0 else label for label in encoding.labels]
        return encoding
