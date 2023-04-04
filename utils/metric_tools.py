
class DSTMetrics:

    def __init__(self, tokenizer, preds, labels):
        self.tokenizer = tokenizer
        self.preds, self.labels = self.decode_ids(preds, labels)

    def compute_joint_goal_accuracy(self, preds, labels):
        """
    
        """
    
        for pred, label in zip(preds, labels):
            break

    def decode_ids(self, preds, labels):
        preds = self.tokenizer.decode(preds, skip_special_tokens=True)  # is there clean_up
        labels = self.tokenizer.decode(labels, skip_special_tokens=True)  # is there clean_up
        return preds, labels


    #TODO: Move decoding and generation to another class where I can tokenize?
    #def generate_state(self, encoding, states_len):

    #    input_ids = encoding['input_ids']
    #    attention_mask = encoding['attention_mask']
    #    self.model.eval()
    #    with torch.no_grad():
    #        generated_ids = self.model.generate(input_ids=input_ids,
    #                                            attention_mask=attention_mask,
    #                                            max_length=states_len,
    #                                            truncation=True,
    #                                            num_beams=beam_search,
    #                                            repetition_penalty=repetition_penalty,
    #                                            length_penalty=1.0,
    #                                            early_stopping=True)

    #        prediction = [self.tokenizer.decode(state, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]


    ## TODO: LAST STEP, THE GOAL HERE IS TO PASS SOME DIALOGUE AND GENERATE STATES
    ##TODO: From text work in progress, see previous method
    #def encode_stage(self, dialogue_hist, states_len=256, beam_search=2, repetition_penalty=2.5):
    #    #TODO: Tokenize dialogue history to pass input and mask to generate state method
    #    #encoding = ...
    #    self.generate_state(encoding, states_len, beam_search, repetition_penalty)

