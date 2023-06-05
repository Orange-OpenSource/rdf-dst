# github.com/snowblink14/smatch
# https://github.com/mdtux89/amr-evaluation
# https://github.com/IBM/transition-amr-parser
from dataclasses import dataclass
from utils.postprocessing import clean_node
import random


@dataclass
class PreDataCollator:
    
    def __init__(self, tokenizer, source_len, target_len, exp_setup, cut_context):

        self.cut_context = cut_context
        self.exp_setup = exp_setup
        self.source_len = source_len
        self.target_len = target_len
        self.user_tkn = '<user_tkn>'
        self.sys_tkn = '<sys_tkn>'
        self.state_tkn = '<state_tkn>'

        self.subject_tkn = '<subject_tkn>'
        self.relation_tkn = '<relation_tkn>'
        self.object_tkn = '<object_tkn>'
        sentinel_tkns = {"additional_special_tokens": [self.user_tkn, self.sys_tkn, self.state_tkn, self.subject_tkn, self.object_tkn, self.relation_tkn]}
        tokenizer.add_special_tokens(sentinel_tkns)
        self.tokenizer = tokenizer

    def __call__(self, batch):
        
        input_ids = []
        attention_mask = []
        labels = []
        dialogue_ids = []
        turn_number = []

        for id, dialogue, states in zip(batch['dialogue_id'], batch['turns'], batch['states']):  # dict, history is a str that is the key
            txt_input, label_rdf = self.create_inputs_outputs(dialogue, states)

            for turn, (txt, rdf) in enumerate(zip(txt_input, label_rdf), 0):
                tokenized = self.tokenize(txt, rdf)

                input_ids.append(tokenized['input_ids'])
                attention_mask.append(tokenized['attention_mask'])
                labels.append(tokenized['labels'])
                dialogue_ids.append(id)
                turn_number.append(turn)

        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': labels, 'dialogue_id': dialogue_ids, 'turn_number': turn_number}


    @staticmethod
    def flatten_rdf_rep(state):
        flatten_dict = dict()
        for triplet in state['triples']:
            if triplet[0] in flatten_dict:
                flatten_dict[triplet[0]].extend(triplet[1:])
            else:
                flatten_dict[triplet[0]] = triplet[1:]

        #flatten_rep = []
        #for k, values in flatten_dict.items():
        #    flat_rdf = f'{k} is '
        #    for v in values:
        #        flat_rdf += f';{v}'
        #    flatten_rep.append(flat_rdf)
        return flatten_dict

        #return '\n'.join(flatten_rep)

    def explicit_info_injection(self, word, i):
        special_tkn = {0: self.subject_tkn, 1: self.relation_tkn, 2: self.object_tkn}
        return special_tkn[i] + clean_node(word)
    
    def create_inputs_outputs(self, dialogue, states):
        """
        This is where we choose the inputs and the rdf we will predict.
        Since we are using the whole state history and the first state cannot be predicted
        with a previous state, we initialize with an empty state and try to predict current state
        # pretokenizing: https://huggingface.co/learn/nlp-course/chapter6/4
        """

        # we can flatten all of the rdf-states and treat them as strings. But maybe the only last one matters?
        toks = {"user": self.user_tkn, "system": self.sys_tkn}

        #smarter triplet rep? Not for now. comment
        #states = map(lambda state: self.flatten_rdf_rep(state), states)
        #states = list(states)

        states = map(lambda state: [[self.explicit_info_injection(val, i) for i, val in enumerate(triple)] for triple in state['triples']], states)
        # shuffling for augmentation
        #states = map(lambda state: random.sample(state, len(state)), states)
        states = list(states)
        states = [[node for rdf in state for node in rdf] for state in states]

        context = ''
        all_context = []
        if self.exp_setup == 3:
            model_input = list(map(lambda state: [self.state_tkn] + state, states[:-1]))
            model_input.insert(0, [self.state_tkn, ' '])
        else:
            prev_states = list(map(lambda state: [self.state_tkn] + state, states[:-1]))
            for i in range(0, len(dialogue), 2):

                speaker = dialogue[i]['speaker']
                context += toks[speaker] + dialogue[i]['text']

                speaker = dialogue[i+1]['speaker']
                context += toks[speaker] + dialogue[i+1]['text']
                all_context.append(context)

            model_input = [diag.split() for diag in all_context]
            
            if self.exp_setup == 1:
                model_input = model_input[:1] + list(map(list.__add__, model_input[1:], prev_states))

                # cutting context in T5 left to right because the sequence is too long. Our threshold is 
                if self.cut_context:
                    model_input = list(map(self.reduce_context, model_input))

        labels = map(lambda state: ','.join(['|'.join(state[i:i+3]) for i in range(0, len(state), 3)]), states)
        labels = list(labels)

        return model_input, labels


    def reduce_context(self, txt):
        """
        cut context for T5 because context is too long for standard T5
        This has been hardcoded as 525, where we have observed that a list with more than these tokens, breaks the model
        """
        threshold = 525
        if len(txt) > threshold:
            slice_val = len(txt) - threshold
            txt = txt[slice_val:]
        return txt

    def tokenize(self, dialogue : list, rdf : str):
        
        encoding = self.tokenizer(dialogue,
                       is_split_into_words=True,
                       padding='max_length',
                       truncation=True,
                       max_length=self.source_len)
        
        target_encoding = self.tokenizer(rdf, padding='max_length',
                                         is_split_into_words=False,
                                         truncation=True,
                                         max_length=self.target_len)
        
        labels = target_encoding.input_ids
        labels = [-100 if label == self.tokenizer.pad_token_id else label for label in labels]

        encoding['labels'] = labels

        return encoding
