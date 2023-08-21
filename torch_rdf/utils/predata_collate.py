# github.com/snowblink14/smatch
# https://github.com/mdtux89/amr-evaluation
# https://github.com/IBM/transition-amr-parser
from dataclasses import dataclass
from utils.postprocessing import clean_node
from typing import Iterable
import random
import re


@dataclass
class PreDataCollator:
    
    def __init__(self, tokenizer, source_len, target_len, exp_setup, dataset_type, ignore_inter, cut_context):

        data_collation = {"multiwoz": self.create_inputs_outputs_multiwoz,
                          "dstc2": self.create_inputs_outputs_dstc2_sfx, "sfx": self.create_inputs_outputs_dstc2_sfx}
        self.data_collation = data_collation[dataset_type]

        self.dataset_type = dataset_type
        self.cut_context = cut_context
        self.ignore_inter_states = ignore_inter
        self.exp_setup = exp_setup
        self.source_len = source_len
        self.target_len = target_len

        self.subject_tkn = '<subject_tkn>'
        self.relation_tkn = '<relation_tkn>'
        self.object_tkn = '<object_tkn>'

        sentinel_tkns = {"additional_special_tokens": [self.subject_tkn, self.object_tkn, self.relation_tkn]}
        tokenizer.add_special_tokens(sentinel_tkns)
        self.tokenizer = tokenizer

    def __call__(self, batch):
        
        input_ids = []
        attention_mask = []
        labels = []
        dialogue_ids = []
        turn_number = []


        for diag_id, dialogue, states in zip(batch['dialogue_id'], batch['turns'], batch['states']):  # dict, history is a str that is the key
            txt_input, label_rdf = self.data_collation(dialogue, states)

            for turn, (txt, rdf) in enumerate(zip(txt_input, label_rdf), 0):
                tokenized = self.tokenize(txt, rdf)

                input_ids.append(tokenized['input_ids'])
                attention_mask.append(tokenized['attention_mask'])
                labels.append(tokenized['labels'])
                dialogue_ids.append(diag_id)
                turn_number.append(turn)
        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': labels, 'dialogue_id': dialogue_ids, 'turn_number': turn_number}


    def explicit_info_injection(self, word, i):
        special_tkn = {0: self.subject_tkn, 1: self.relation_tkn, 2: self.object_tkn}
        return special_tkn[i] + clean_node(word)
    
    def filter_triples(self, triple):
        randompatternRegex = re.compile(r'\/[a-zA-Z0-9]+')
        if triple[0] == '_:system':
            return False
        for el in triple:
            # ignore rejected searches, results, etc. Intermediate triples that create noise
            if randompatternRegex.search(el):
                return False
        return True
    
    def rearrange_sys_triples(self, states):
        """
        Current implementation has triples from the curr sys utterance, they have to be moved.
        The DST Task is user intent, not system and user intent
        # what are the intents then? https://gitlab.tech.orange/NEPAL/task-oriented-dialogue/poc-rdf/-/blob/master/poc_rdf/dst.py
        ONLY FOR MULTIWOZ, OTHER STATES ARE FINE
        """
        user_raw_states = []
        sys_raw_states = []
        for s in states:
            user_raw_rdf = []
            sys_raw_rdf = []
            for triple in s:
            #    # user's edge can be deny which does express user intent, so we are keeping this.
            #    # Identifying if the user greeted in utterance is important as well. See link above
            #    # This is how user expresses emotions and other info there.
            #    print(triple)

                if triple[0] not in ['_:search', '_:user']:
                    # this is from the curr sys utterance. It should be in the present state
                    sys_raw_rdf.append(triple)
                else:
                    user_raw_rdf.append(triple)
            
            #print("NEXT ONE")
            #print()
            user_raw_states.append(user_raw_rdf)
            sys_raw_states.append(sys_raw_rdf)
        

        clean_state = [user_raw_states[0]]
        #clean_state = []
        for i in range(len(sys_raw_states)-1):
            clean_rdf = user_raw_states[i+1] + sys_raw_states[i]
            clean_state.append(clean_rdf)

        return clean_state
        
    
    def clean_states(self, states):
        states = map(lambda state: [[self.explicit_info_injection(val, i) for i, val in enumerate(triple)] for triple in state], states)
        # shuffling for augmentation: triple order does not matter.
        states = map(lambda state: random.sample(state, len(state)), states)
        states = list(states)
        return [[node for rdf in state for node in rdf] for state in states]
    

    def states_processing(self, states):
        if self.ignore_inter_states:
            states = map(lambda state: list(filter(self.filter_triples, state['triples'])), states)
            states = list(states)
            if self.dataset_type == "multiwoz":
                states = self.rearrange_sys_triples(states)

        else:
            states = [state['triples'] for state in states]
        
        states  = self.clean_states(states)

        linearized_states = map(lambda state: ','.join([';'.join(state[i:i+3]) for i in range(0, len(state), 3)]), states)
        labels = list(linearized_states)
        return {"states": states, "labels": labels}


    def model_input_processing(self, model_input, states, labels):

        if self.exp_setup == 1:
            # non linearized, they are in a list so tokenizer can work with these
            prev_states = list(map(lambda state: ['STATE '] + state, states[:-1]))
            model_input = model_input[:1] + list(map(list.__add__, model_input[1:], prev_states))
        elif self.exp_setup in [3, 4]:
            first_turn = model_input[0]
            prev_states = ['STATE ' + state for state in labels[:-1]]
            model_input = [txt + ' ' + s for txt, s in zip(model_input[1:], prev_states)]
            model_input.insert(0, first_turn)

        # cutting context in T5 left to right because the sequence is too long.

        if self.cut_context and self.exp_setup == 1:
            model_input = list(map(self.reduce_context, model_input))

        return model_input


    def create_inputs_outputs_dstc2_sfx(self, dialogue, states):
        """
        This is where we choose the inputs and the rdf we will predict.
        Since we are using the whole state history and the first state cannot be predicted
        with a previous state, we initialize with an empty state and try to predict current state
        # pretokenizing: https://huggingface.co/learn/nlp-course/chapter6/4
        """

        toks = {"user": 'USER ', "system": "SYSTEM "}

        context = ''

        model_input = []
        # removing system triples, user and states that pollute generation
        processed_states = self.states_processing
        states = processed_states['states']
        labels = processed_states['labels']
        for i in range(0, len(dialogue), 2):

            # SYS UTTERANCE
            sys_speaker = dialogue[i]['speaker']
            sys_utterance = toks[sys_speaker] + dialogue[i]['text']

            usr_speaker = dialogue[i+1]['speaker']
            usr_utterance = toks[usr_speaker] + dialogue[i-1]['text']

            curr_turn_input = sys_utterance + ' ' + usr_utterance

            if self.exp_setup in [1, 2]:
                context += (curr_turn_input + ' ')
            
            
            if self.exp_setup in [4, 5]:
                curr_turn_input = curr_turn_input.strip()
            elif self.exp_setup in [1, 2]:
                curr_turn_input = (context + curr_turn_input).strip()
                if self.exp_setup == 1:
                    curr_turn_input = curr_turn_input.split()

            model_input.append(curr_turn_input)

        model_input = self.model_input_processing(model_input, states, labels)
        
        return model_input, labels

    def create_inputs_outputs_multiwoz(self, dialogue, states):
        """
        This is where we choose the inputs and the rdf we will predict.
        Since we are using the whole state history and the first state cannot be predicted
        with a previous state, we initialize with an empty state and try to predict current state
        # pretokenizing: https://huggingface.co/learn/nlp-course/chapter6/4
        """

        toks = {"user": 'USER ', "system": "SYSTEM "}

        context = ''

        model_input = []
        # removing system triples, user and states that pollute generation
        processed_states = self.states_processing
        states = processed_states['states']
        labels = processed_states['labels']
        for i in range(0, len(dialogue), 2):

            # USER UTTERANCE
            usr_speaker = dialogue[i]['speaker']
            curr_turn_usr = toks[usr_speaker] + dialogue[i]['text']
            prev_turn_sys = ''
            if i > 0:
                # prev sys utterance
                sys_speaker = dialogue[i+1]['speaker']
                prev_turn_sys = toks[sys_speaker] + dialogue[i-1]['text']

                if self.exp_setup in [1, 2]:
                    # prev user utterance
                    prev_turn_user = toks[usr_speaker] + dialogue[i-2]['text']
                    context += (prev_turn_user + ' ' + prev_turn_sys + ' ')
            
            
            if self.exp_setup in [4, 5]:
                curr_turn_input = curr_turn_usr.strip()
            elif self.exp_setup in [1, 2]:
                curr_turn_input = (context + curr_turn_usr).strip()
                if self.exp_setup == 1:
                    curr_turn_input = curr_turn_input.split()


            model_input.append(curr_turn_input)

        model_input = self.model_input_processing(model_input, states, labels)
        
        return model_input, labels


    def reduce_context(self, txt):
        """
        cut context for T5 because context is too long for standard T5
        This has been hardcoded as 525, where we have observed that a list with more than these tokens, breaks the model
        """
        threshold = self.tokenizer.model_max_length // 2  # 512 usually
        if len(txt) > threshold:
            slice_val = len(txt) - threshold
            txt = txt[slice_val:]
        return txt

    def tokenize(self, dialogue : Iterable, rdf : str):
        
        if self.exp_setup == 1:
            encoding = self.tokenizer(dialogue,
                           is_split_into_words=True,
                           padding='max_length',
                           truncation=True,
                           max_length=self.source_len)
        else:
            encoding = self.tokenizer(dialogue,
                           is_split_into_words=False,
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
