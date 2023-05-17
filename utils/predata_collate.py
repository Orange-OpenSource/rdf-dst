from dataclasses import dataclass
import re


@dataclass
class PreDataCollator:
    
    def __init__(self, tokenizer, source_len, target_len, exp_setup):

        self.exp_setup = exp_setup
        self.source_len = source_len
        self.target_len = target_len
        self.user_tkn = '<user_tkn>'
        self.sys_tkn = '<sys_tkn>'
        self.state_tkn = '<state_tkn>'
        sentinel_tkns = {"additional_special_tokens": [self.user_tkn, self.sys_tkn, self.state_tkn]}
        tokenizer.add_special_tokens(sentinel_tkns)
        self.tokenizer = tokenizer

    def __call__(self, batch):
        
        input_ids = []
        attention_mask = []
        labels = []
        dialogue_ids = []
        turn_number = []
        all_states = []
        
        for id, dialogue, states in zip(batch['dialogue_id'], batch['turns'], batch['states']):  # dict, history is a str that is the key
            txt_input, label_rdf = self.create_inputs_outputs(dialogue, states)

            for turn, (txt, rdf) in enumerate(zip(txt_input, label_rdf), 0):
                tokenized = self.tokenize(txt, rdf)
                input_ids.append(tokenized['input_ids'])
                attention_mask.append(tokenized['attention_mask'])
                labels.append(tokenized['labels'])
                dialogue_ids.append(id)
                turn_number.append(turn)

                all_states.append(states[turn]['triples'])


        return {'input_ids': input_ids, 'attention_mask': attention_mask, "states": all_states,
                'labels': labels, 'dialogue_id': dialogue_ids, 'turn_number': turn_number}


    @staticmethod
    def flatten_rdf_rep(state):
        #state = (val.strip() for triplet in state['triples'] for val in triplet)
        #state = [val.replace('_:' , '') for triplet in state for val in triplet]
        flatten_dict = dict()
        for triplet in state['triples']:
            triplet = [val.strip() for val in triplet]
            triplet = [val.replace('_:', '') for val in triplet]
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

    def create_inputs_outputs(self, dialogue, states):
        """
        This is where we choose the inputs and the rdf we will predict.
        Since we are using the whole state history and the first state cannot be predicted
        with a previous state, we initialize with an empty state and try to predict current state
        """

        # we can flatten all of the rdf-states and treat them as strings. But maybe the only last one matters?
        toks = {"user": self.user_tkn, "system": self.sys_tkn}

        states = map(lambda state: ','.join([val.strip() for triplet in state['triples'] for val in triplet]), states)
        states = list(states)

        #smarter triplet rep? Not for now. comment
        #states = map(lambda state: self.flatten_rdf_rep(state), states)
        #states = list(states)


        dialogue_context = []
        context = ''
        if self.exp_setup == 3:
            input_txt = [''].extend(states[1:])
        else:
            for i in range(0, len(dialogue), 2):

                speaker = dialogue[i]['speaker']
                context += toks[speaker] + dialogue[i]['text'] + toks[speaker]
                speaker = dialogue[i+1]['speaker']
                context += toks[speaker] + dialogue[i+1]['text'] + toks[speaker]
                dialogue_context.append(context)
            prev_states = list(map(lambda state: self.state_tkn + state + self.state_tkn, states[:-1]))
            input_txt = dialogue_context[:1] + [diag + prev_states[i] for i, diag in enumerate(dialogue_context[1:])] if self.exp_setup == 1 else dialogue_context

        return input_txt, states

    def tokenize(self, dialogue : str, rdf : str):
        
        encoding = self.tokenizer(dialogue,
                       #is_split_into_words=True,
                       padding='max_length',
                       truncation=True,
                       max_length=self.source_len)
        
        target_encoding = self.tokenizer(rdf, padding='max_length',
                                         truncation=True,
                                         max_length=self.target_len)
        
        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        encoding['labels'] = labels

        return encoding

