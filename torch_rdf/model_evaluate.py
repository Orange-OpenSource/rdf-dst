# Copyright 2023 Orange
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Software Name : knowledge-graph-dst
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: H. Andres Gonzalez


from dotenv import load_dotenv
load_dotenv()  # load keys and especially w and biases to see visualizations. Looking in curr dir

import re
import os
import glob
# longt5 needs special module to avoid errors
from transformers import AutoTokenizer, T5ForConditionalGeneration, LongT5ForConditionalGeneration
from utils.data_loader import DialogueRDFData
from utils.args import create_arg_parser
from utils.metric_tools import DSTMetrics
from evaluator import MyEvaluation
from utils.predata_collate import PreDataCollator
from peft import PeftModel, PeftConfig

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes

def preprocessing(collator, dataset, num_workers, batch_size, method):

    data = DialogueRDFData(collator, num_workers=num_workers,
                           dataset=dataset,
                           batch_size=batch_size,
                           inference_time=True)
    data.load_hf_data(method)
    dataloaders = data.create_loaders(subsetting=subsetting)

    return {'test': dataloaders['test']}

def evaluating(model, tokenizer, test_dataloader, device, 
             target_len, dst_metrics, beam_size, path):


    logging.info("Inference stage")


    my_evaluation = MyEvaluation(model, tokenizer, device, target_len, dst_metrics, beam_size, path=path)
    my_evaluation(test_dataloader, validation=False, verbose=True)
    print(my_evaluation.results)

def load_model(model_max_len, model_path, file_path, peft):

    ckpt_path = find_version_num(file_path, peft)
    #ckpt_path = '../results/models/tb_logs/flan-t5_experiment_1/version_0/checkpoints/best_dst_ckpt/'

    if peft:
        peft_model_id = ckpt_path
        config = PeftConfig.from_pretrained(peft_model_id)
        if 'long' not in file_path:
            model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            model = LongT5ForConditionalGeneration.from_pretrained(model_path)
            #model = LongT5ForConditionalGeneration.from_pretrained(ckpt_path)

        model = PeftModel.from_pretrained(model, peft_model_id)
        if peft != 'prefix':
            model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(model_path, extra_ids=0, truncation=True, truncation_side='left',
                                                  model_max_length=model_max_len) 
    else:
        if 'long' not in file_path:
            model = T5ForConditionalGeneration.from_pretrained(ckpt_path)
        else:
            model = LongT5ForConditionalGeneration.from_pretrained(ckpt_path)

        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, extra_ids=0, truncation=True, truncation_side='left',
                                                  model_max_length=model_max_len) 

    store_path = os.path.dirname(ckpt_path)
    return {"model": model, "tokenizer": tokenizer, "store_path": store_path}


def find_version_num(path, peft):

    dirs = [d for d in os.listdir(path) if 'checkpoints' in os.listdir(os.path.join(path, d))]
    assert dirs, "No version has any checkpoints. Did you train the model?"
    newest_version = max(map(regex_match, dirs))
    parent_dir = os.path.join(path, f"version_{newest_version}", "checkpoints", peft)
    pattern = os.path.join(parent_dir, "best_dst_ckpt")
    checkpoints = [dir_path for dir_path in glob.glob(pattern) if os.path.isdir(dir_path)]
    return max(checkpoints, key=os.path.getctime)

def regex_match(dir_name):
    versionRegex = re.compile(r"^version_(\d+)$")
    res_match = versionRegex.search(dir_name)
    if res_match:
        return int(res_match.group(1))
    else:
        return -1

def main():

    global subsetting
    global store

    args = create_arg_parser()
    models = {'t5': 't5', 'flan-t5': 'google/flan-t5', 'long-t5-local': 'google/long-t5-local', 'long-t5-tglobal': 'google/long-t5-tglobal'}
    dataset = args.dataset
    num_workers = args.num_workers
    batch_size = args.batch
    method = args.method
    device = args.device
    model_name = models[args.model]
    beam_size = args.beam
    peft_type = args.peft
    model_path =  model_name + '-' + args.model_size

    bool_4_args = {"no": False, "yes": True}

    subsetting = bool_4_args[args.subsetting]
    store = bool_4_args[args.store_output]
    ignore_inter = bool_4_args[args.ig]


    length_exp_setup = {1: {"source_len": 1024, "target_len": 1024, "setup": "user, context and states"},
                        2: {"source_len": 512,  "target_len": 1024, "setup": "user and context"},
                        3: {"source_len": 1024, "target_len": 1024, "setup": "user, prev. sys input and states"},
                        4: {"source_len": 768,  "target_len": 1024, "setup": "user and states"},
                        5: {"source_len": 256,  "target_len": 1024, "setup": "user input"},
                        6: {"source_len": 1024, "target_len": 1024, "setup": "only states"}
                        }
    if args.experimental_setup in [4, 5, 6]:
        logging.warning(f"YOU ARE RUNNING ABLATION NUMBER {args.experimental_setup - 3}")

    experimental_setup = args.experimental_setup

    source_len = length_exp_setup[experimental_setup]["source_len"]
    target_len = length_exp_setup[experimental_setup]["target_len"]
    tokenizer_max_length = max([target_len, source_len])
    #source_len = source_len * 2 if ((model_name[:2] != 't5') and (experimental_setup == 1)) else source_len
    if not peft_type:
        peft_type = ''
    model_checkpoint_name = f"{model_name}_{args.model_size}_experiment_{experimental_setup}"

    if os.getenv('DPR_JOB'):
        path = os.path.join("/userstorage/", os.getenv('DPR_JOB'))
    else:
        path = "."

    model_checkpoint_name = model_checkpoint_name.replace('google/', '')
    file_path = os.path.join(path, 'tb_logs', model_checkpoint_name)

    loaded_config = load_model(tokenizer_max_length, model_path, file_path, peft_type)
    tokenizer = loaded_config["tokenizer"]
    model = loaded_config["model"]
    store_path = loaded_config["store_path"]
    if not os.path.exists(store_path):
        raise Exception("No path found to store outputs")
    store_path = os.path.join(store_path, dataset)

    if not os.path.exists(store_path):
        os.makedirs(store_path)


    cut_context = True if experimental_setup == 1 else False
    collator = PreDataCollator(tokenizer, source_len, target_len, experimental_setup, dataset_type=dataset,
                               ignore_inter=ignore_inter, cut_context=cut_context)
    dataloaders = preprocessing(collator, dataset, num_workers, batch_size, method)


    dst_metrics = DSTMetrics()  # this is loading the metrics now so we don't have to do this again

    logging.info(f"Outputs will be stored in\n{store_path}")
    evaluating(model, tokenizer, dataloaders['test'], device, 
               target_len, dst_metrics, beam_size, path=store_path)

if __name__ == '__main__':
    main()
