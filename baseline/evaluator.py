# Copyright (c) 2023 Orange

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITEDTOTHE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Software Name : knowledge-graph-dst
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: H. Andres Gonzalez

import pandas as pd
import torch
import os
from utils.postprocessing import postprocess_states
from tqdm import tqdm
from accelerate import Accelerator
from transformers.generation import GenerationConfig
from codecarbon import track_emissions

import logging

logging.basicConfig(level=logging.INFO)

SEED = 42  # for replication purposes
accelerator = Accelerator()


class MyEvaluation:

    def __init__(self, model, tokenizer, device, target_length, dst_metrics, beam_size=0, path=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dst_metrics = dst_metrics
        self.results = {}
        self.path = path
    
        self.gen_kwargs = {"max_new_tokens": target_length,
                           "min_new_tokens": target_length // 4,
                           "early_stopping": True,
                           "num_beams": beam_size
                          }
        if beam_size <= 0:
            beam_size = 'greedy'
            self.gen_kwargs.pop("num_beams")
        self.beam_size = beam_size

    @track_emissions(project_name='dst-base', save_to_api=False, country_iso_code='FRA', offline=True,
                     experiment_id='real_eval_baseline-2-base-dst-full', output_file='real_eval_base_2_basefull_emissions.csv')
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
    
                if validation:
                    model_outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                    # loss, logits, encoder_last_hidden_state, past_key_values
                    loss = model_outputs.loss
                    total_loss += loss.item()
    
                else:
                    step_output = self.generate_rdfs(inputs, attention_mask, labels, dialogue_ids)
                    outputs.append(step_output)

                #elif validation and ((step != 0) and (step % eval_steps == 0)):
                #    step_output = self.generate_rdfs(inputs, attention_mask, labels, dialogue_ids)
                #    outputs.append(step_output)
                    
    
        if validation:
            return total_loss / len(eval_data)
        elif not validation and self.path:
            self.store_outputs(outputs)
            self.results = self.evaluate_outputs(outputs)

        outputs.clear()


    def evaluate_outputs(self, outputs):
        return self.dst_metrics(outputs)

    def store_outputs(self, outputs):
        states_df = pd.DataFrame(outputs)
        states_df.to_csv(os.path.join(self.path, f"outputs_beam_{self.beam_size}.csv"), index=False)
    


    def generate_rdfs(self, input_ids, attention_mask, labels, dialogue_ids):
        
        '''
        Generation is expensive. Saving up gpu and using cpu instead!
        '''

        input_ids = input_ids.detach()
        attention_mask = attention_mask.detach()

        generation_sizes = [torch.sum(lab[0] != -100).item() for lab in labels.split(1)]  # split across first dim, getting rows, must index actual tensor
        self.gen_kwargs["max_new_tokens"] = max(generation_sizes)
        self.gen_kwargs["min_new_tokens"] = min(generation_sizes)

        gen_cfg = GenerationConfig.from_dict(self.gen_kwargs)

        generated_tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=gen_cfg)

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

