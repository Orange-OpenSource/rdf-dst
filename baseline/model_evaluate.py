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

from dotenv import load_dotenv
load_dotenv()  # load keys and especially w and biases to see visualizations. Looking in curr dir

import re
import os
import glob
# longt5 needs special module to avoid errors
from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.data_loader import DialogueData
from utils.args import create_arg_parser
from utils.metric_tools import DSTMetrics
from evaluator import MyEvaluation
from utils.predata_collate import BaselinePreDataCollator
from peft import PeftModel, PeftConfig

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes

def preprocessing(collator, dataset, num_workers, batch_size, method):

    data = DialogueData(collator, num_workers=num_workers,
                        dataset=dataset,
                        batch_size=batch_size,
                        inference_time=True)

    data.load_hf_data(method)
    dataloaders = data.create_loaders(subsetting=subsetting)

    return {'test': dataloaders["test"]}


def load_model(tok_max_len, model_path, file_path, peft):

    ckpt_path = find_version_num(file_path, peft)
    #ckpt_path = '../results/models/tb_logs/flan-t5_experiment_1/version_0/checkpoints/best_dst_ckpt/'
    #ckpt_path = '/data/userstorage/yddb9991/base-flant5-dst/baseline/tb_logs/t5_base_experiment_2/full/checkpoints/best_dst_ckpt' 
    if peft:
        peft_model_id = ckpt_path
        # ONLY VALID IF PATH LOADED FROM IS THE SAME AS THE PATH IT STORES THE MODEL AFTER TRAINING
        # otherwise change adapter config json
        config = PeftConfig.from_pretrained(peft_model_id)
        model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, extra_ids=0, truncation_side='left',
                                                  truncation=True, model_max_length=tok_max_len) 
        # TEMPORARY SOLUTION...
        #model = T5ForConditionalGeneration.from_pretrained(model_path)
        #model = PeftModel.from_pretrained(model, peft_model_id)
        #tokenizer = AutoTokenizer.from_pretrained(model_path) 
        if peft != 'prefix':
            model = model.merge_and_unload()

    else:
        model = T5ForConditionalGeneration.from_pretrained(ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, extra_ids=0, truncation_side='left',
                                                  truncation=True, model_max_length=tok_max_len) 

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



def evaluate(model, tokenizer, test_dataloader, device, 
             target_len, dst_metrics, beam_size, path):

    logging.info("Inference stage")

    my_evaluation = MyEvaluation(model, tokenizer, device, target_len, dst_metrics, beam_size, path=path)
    my_evaluation(test_dataloader, validation=False, verbose=True)
    print(my_evaluation.results)


def main():

    global subsetting
    global store

    args = create_arg_parser()
    model_name = args.model
    dataset = args.dataset
    num_workers = args.num_workers
    batch_size = args.batch
    method = args.method
    device = args.device
    beam_size = args.beam
    peft_type = args.peft
    model_path =  model_name + '-' + args.model_size

    bool_4_args = {"no": False, "yes": True}
    length_exp_setup = {1: {"source_len": 512,  "target_len": 256, "setup": "user, context and states"},  # 1024?
                        2: {"source_len": 512,  "target_len": 256, "setup": "user and context"},
                        3: {"source_len": 512,  "target_len": 256, "setup": "user, prev. sys input and states"},
                        4: {"source_len": 512,  "target_len": 256, "setup": "user and states"},
                        5: {"source_len": 256,  "target_len": 256, "setup": "user input"},
                        6: {"source_len": 256,  "target_len": 256, "setup": "only states"}
                        }
    
    if args.experimental_setup in [4, 5, 6]:
        logging.warning(f"YOU ARE RUNNING ABLATION NUMBER {args.experimental_setup - 3}")

    subsetting = bool_4_args[args.subsetting]
    store = bool_4_args[args.store_output]
    include_prev_sys = bool_4_args[args.incl]

    experimental_setup = args.experimental_setup
    source_len = length_exp_setup[experimental_setup]["source_len"]

    target_len = length_exp_setup[experimental_setup]["target_len"]
    tok_max_len = max([target_len, source_len])
    if not peft_type:
        peft_type = ''
    model_checkpoint_name = f"{model_name}_{args.model_size}_experiment_{experimental_setup}"

    if os.getenv('DPR_JOB'):
        path = os.path.join("/userstorage/", os.getenv('DPR_JOB'))
    else:
        path = "."

    file_path = os.path.join(path, 'tb_logs', model_checkpoint_name)

    loaded_config = load_model(tok_max_len, model_path, file_path, peft_type)
    tokenizer = loaded_config["tokenizer"]
    model = loaded_config["model"]

    store_path = loaded_config["store_path"]
    if not os.path.exists(store_path):
        raise Exception("No path found to store outputs")
    store_path = os.path.join(store_path, dataset)

    if not os.path.exists(store_path):
        os.makedirs(store_path)


    collator = BaselinePreDataCollator(tokenizer, source_len, target_len, experimental_setup,
                                       dataset_type=dataset, include_prev_sys=include_prev_sys)

    model.resize_token_embeddings(len(tokenizer))
    dataloaders = preprocessing(collator, dataset, num_workers, batch_size, method)

    dst_metrics = DSTMetrics()  # this is loading the metrics now so we don't have to do this again

    logging.info(f"Outputs will be stored in\n{store_path}")
    evaluate(model, tokenizer, dataloaders['test'], device, 
             target_len, dst_metrics, beam_size, path=store_path)

if __name__ == '__main__':
    main()
