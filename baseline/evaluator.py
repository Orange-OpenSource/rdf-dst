import pandas as pd
import torch
import os
from utils.postprocessing import postprocess_states
from tqdm import tqdm
from accelerate import Accelerator

import logging

logging.basicConfig(level=logging.INFO)

SEED = 42  # for replication purposes
accelerator = Accelerator()


class MyEvaluation:

    def __init__(self, model, tokenizer, device, target_length, dst_metrics, path=None, is_peft=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dst_metrics = dst_metrics
        self.results = {}
        self.path = path
        self.is_peft = is_peft
        self.gen_kwargs = {"max_length": target_length,
                           "min_length": target_length,
                           "early_stopping": True
                          }
    

    def __call__(self, eval_data, eval_steps=None, validation=False, verbose=False):

        eval_data, self.model = accelerator.prepare(eval_data, self.model)
    
        self.model.eval()
    
        total_loss = 0
        outputs = []
        disable = not verbose
        with torch.no_grad():
            for step, batch in tqdm(enumerate(eval_data), disable=disable):
    
                inputs = batch['input_ids']
                labels = batch['labels']
                attention_mask = batch['attention_mask']
                dialogue_ids = batch['dialogue_id']  # this is a list no need to move to cpu
    
                model_outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                # loss, logits, encoder_last_hidden_state, past_key_values
                loss = model_outputs.loss
                total_loss += loss.item()
    
                if not validation:
                    step_output = self.generate_rdfs(inputs, attention_mask, labels, dialogue_ids)
                    outputs.append(step_output)

                #elif validation and ((step != 0) and (step % eval_steps == 0)):
                #    step_output = self.generate_rdfs(inputs, attention_mask, labels, dialogue_ids)
                #    outputs.append(step_output)
                    
    
        if not validation and self.path:
            self.store_outputs(outputs)
            self.results = self.evaluate_outputs(outputs)

        total_loss /= len(eval_data)
        outputs.clear()

        return total_loss

    def evaluate_outputs(self, outputs):
        return self.dst_metrics(outputs)

    def store_outputs(self, outputs):
        states_df = pd.DataFrame(outputs)
        states_df.to_csv(os.path.join(self.path, "outputs.csv"), index=False)
    


    def generate_rdfs(self, input_ids, attention_mask, labels, dialogue_ids):
        
        '''
        Generation is expensive. Saving up gpu and using cpu instead!
        '''

        input_ids = input_ids.detach()
        attention_mask = attention_mask.detach()
        if self.is_peft:
            peft_model = self.model
            self.gen_kwargs.update({"input_ids": input_ids, "attention_mask": attention_mask})
            generated_tokens = peft_model.generate(**self.gen_kwargs)
        else:
            generated_tokens = self.model.generate(input_ids, attention_mask=attention_mask, **self.gen_kwargs)

        input_ids = input_ids
        decoded_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        generated_tokens = generated_tokens.detach()
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
        labels = torch.where(labels != -100, labels, 0)
        labels = labels.detach()
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        decoded_labels = [postprocess_states(label) for label in decoded_labels]
        decoded_preds = [postprocess_states(pred) for pred in decoded_preds]

        if isinstance(dialogue_ids, list):
            dialogue_ids = dialogue_ids
        elif torch.tensor(dialogue_ids):
            dialogue_ids = dialogue_ids.detach()
    

        return {"preds": decoded_preds, "labels": decoded_labels,
                "inputs": decoded_inputs, "ids": dialogue_ids}
