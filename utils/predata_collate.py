from dataclasses import dataclass
import re


@dataclass
class PreDataCollator:
    
    def __init__(self, tokenizer, source_len, target_len):

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
        
        for id, dialogue, states in zip(batch['dialogue_id'], batch['turns'], batch['states']):  # dict, history is a str that is the key
            txt_input, label_rdf = self.create_inputs(dialogue, states)

            # current turn ids have chars in their ID, removing to keep only number
            #idRegex = re.compile(r"(\d+)(.json$)?")
            #mo = idRegex.search(id)
            #if mo:
            #    id = int(mo.group(1))  # ints take less memory
            #else:
            #    raise Exception(f"Regex match failed, dialogue ids may be incorrect after preprocessing. THIS IS THE ID: {id}")

            for turn, (txt, rdf) in enumerate(zip(txt_input, label_rdf), 1):
                tokenized = self.tokenize(txt, rdf)
                input_ids.append(tokenized['input_ids'])
                attention_mask.append(tokenized['attention_mask'])
                labels.append(tokenized['labels'])
                dialogue_ids.append(id)
                turn_number.append(turn)


        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': labels, 'dialogue_id': dialogue_ids, 'turn_number': turn_number}


    def create_inputs(self, dialogue, states):
        """
        This is where we choose the inputs and the rdf we will predict.
        Since we are using the whole state history and the first state cannot be predicted
        with a previous state, we initialize with an empty state and try to predict current state
        """

        # we can flatten all of the rdf-states and treat them as strings. But maybe the only last one matters?
        toks = {"user": self.user_tkn, "system": self.sys_tkn}

        txt = ''
        flattened_rdfs = []
        txt_input = []

        curr_rdf = states[0]
        flat_curr = ','.join([val.strip() for triplet in curr_rdf['triples'] for val in triplet])
        flattened_rdfs.append(flat_curr)

        for i in range(0, len(dialogue), 2):
            speaker = dialogue[i]['speaker']
            txt += toks[speaker] + dialogue[i]['text'] + toks[speaker]
            speaker = dialogue[i+1]['speaker']
            txt += toks[speaker] + dialogue[i+1]['text'] + toks[speaker]

            if i > 0:
                # states are half of turns so divide by 2 to get idx. This already skips first txt with empty previous rdf!
                idx = i // 2
                prev_rdf = states[idx-1]
                flat_prev = ','.join([val.strip() for triplet in prev_rdf['triples'] for val in triplet])
                txt += self.state_tkn + flat_prev + self.state_tkn 

                curr_rdf = states[idx]
                flat_curr = ','.join([val.strip() for triplet in curr_rdf['triples'] for val in triplet])
                flattened_rdfs.append(flat_curr)

            txt_input.append(txt)
            txt = ''


        return txt_input, flattened_rdfs

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
