from dataclasses import dataclass
from utils.postprocessing import clean_node
import random


@dataclass
class PreDataCollator:
    
    def __init__(self, tokenizer, source_len, target_len, exp_setup):

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

        #all_states = []
        
        for id, dialogue, states in zip(batch['dialogue_id'], batch['turns'], batch['states']):  # dict, history is a str that is the key
            txt_input, label_rdf = self.create_inputs_outputs(dialogue, states)

            for turn, (txt, rdf) in enumerate(zip(txt_input, label_rdf), 0):
                tokenized = self.tokenize(txt, rdf)
                input_ids.append(tokenized['input_ids'])
                attention_mask.append(tokenized['attention_mask'])
                labels.append(tokenized['labels'])
                dialogue_ids.append(id)
                turn_number.append(turn)

                #all_states.append(states[turn]['triples'])


        return {'input_ids': input_ids, 'attention_mask': attention_mask,# "states": all_states
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
        """

        # we can flatten all of the rdf-states and treat them as strings. But maybe the only last one matters?
        toks = {"user": self.user_tkn, "system": self.sys_tkn}

        #smarter triplet rep? Not for now. comment
        #states = map(lambda state: self.flatten_rdf_rep(state), states)
        #states = list(states)

        states = map(lambda state: [clean_node(val) for triplet in state['triples'] for val in triplet], states)
        #states = map(lambda state: [self.explicit_info_injection(val, i) for triplet in state['triples'] for i, val in enumerate(triplet)], states)
        states = map(lambda state: ['|'.join(state[i:i+3]) for i in range(0, len(state), 3)], states)

        #states = map(lambda state: ','.join(random.sample(state, len(state))), states)
        # no shuffling
        states = map(lambda state: ','.join(state), states)
        states = list(states)

        dialogue_context = []
        #context = ''
        context = []
        if self.exp_setup == 3:
            input_txt = [self.state_tkn + ' ' + self.state_tkn] + list(map(lambda state: self.state_tkn + state + self.state_tkn, states[:-1]))
        else:
            for i in range(0, len(dialogue), 2):

                speaker = dialogue[i]['speaker']
                #context += toks[speaker] + dialogue[i]['text'] + toks[speaker]
                diag = dialogue[i]['text'].split()
                context.append(toks[speaker])
                context.extend(diag)
                context.append(toks[speaker])

                speaker = dialogue[i+1]['speaker']
                diag = dialogue[i+1]['text'].split()
                context.append(toks[speaker])
                context.extend(diag)
                context.append(toks[speaker])
                #context += toks[speaker] + dialogue[i+1]['text'] + toks[speaker]
                dialogue_context.append(context)
            prev_states = list(map(lambda state: self.state_tkn + state + self.state_tkn, states[:-1]))

            #if self.exp_setup == 1:
            #    counting_diag = [diag.replace(self.user_tkn, '') for diag in dialogue_context]
            #    counting_diag = [diag.replace(self.sys_tkn, '').split() for diag in counting_diag]
            #    #diag_size = len([tok for turn in counting_diag for tok in turn])
            #    diag_size = [len(diag) for diag in counting_diag]
            #    state_size = [len(state.split(',')) for state in states]

                #last_state = states[-1].split(',')
                #if len(last_state) > 40:
                #    print(len(last_state))
                #    print()
                #    print(diag_size)
                #    print()
                #    raise SystemExit

            print()
            print(dialogue_context[3])
            print()
            print(prev_states[3])
            print()
            print()
            raise SystemExit
            input_txt = dialogue_context[:1] + [diag + prev_states[i] for i, diag in enumerate(dialogue_context[1:])] if self.exp_setup == 1 else dialogue_context
            #print()
            #print(input_txt)
            #print()
            #raise SystemExit
        
        return input_txt, states

    def tokenize(self, dialogue : str, rdf : str):
        
        print()
        print(dialogue)
        print()
        raise SystemExit

        encoding = self.tokenizer(dialogue,
                       #is_split_into_words=True,
                       padding='max_length',
                       truncation=True,
                       max_length=self.source_len)
        
        target_encoding = self.tokenizer(rdf, padding='max_length',
                                         truncation=True,
                                         max_length=self.target_len)
        
        labels = target_encoding.input_ids
        labels = [-100 if label == self.tokenizer.pad_token_id else label for label in labels]

        encoding['labels'] = labels

        return encoding
