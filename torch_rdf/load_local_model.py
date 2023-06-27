from dotenv import load_dotenv
load_dotenv()  # load keys and especially w and biases to see visualizations. Looking in curr dir

import wandb
import math
import re
import os
import glob
# longt5 needs special module to avoid errors
from transformers import AutoTokenizer, T5ForConditionalGeneration, LongT5ForConditionalGeneration
from utils.data_loader import DialogueData
from utils.args import create_arg_parser
from utils.metric_tools import DSTMetrics
from trainer import MyTrainer, MyEvaluation
from utils.predata_collate import BaselinePreDataCollator
from torch.utils.tensorboard import SummaryWriter

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes


def evaluate(model, tokenizer, test_dataloader, device, 
             target_len, dst_metrics):


    logging.info("Inference stage")


    my_evaluation = MyEvaluation(model, tokenizer, device, target_len, dst_metrics)
    my_evaluation(test_dataloader, validation=False, verbose=True)
    print(my_evaluation.results)

def load_model(name):
    file_path = f'./tb_logs/{name}/'
    ckpt_path = find_version_num(file_path)
    model = T5ForConditionalGeneration.from_pretrained(ckpt_path)
    #tokenizer = AutoTokenizer.from_pretrained(model_name, extra_ids=0) 
    return model


def find_version_num(path):


    dirs = [d for d in os.listdir(path) if 'checkpoints' in os.listdir(path + d)]
    assert dirs, "No version has any checkpoints. Did you train the model?"
    newest_version = max(map(regex_match, dirs))
    parent_dir = f"{path}version_{newest_version}/checkpoints"
    pattern = parent_dir + "/best_dst_ckpt-*"
    checkpoints = [dir_path for dir_path in glob.glob(pattern) if os.path.isdir(dir_path)]
    return max(checkpoints, key=os.path.getctime)

def regex_match(dir_name):
    versionRegex = re.compile(r"^version_(\d+)$")
    res_match = versionRegex.search(dir_name)
    return int(res_match.group(1))

def main():

    global subsetting
    global store

    args = create_arg_parser()
    # self attention layer swapped with local attention or transient-global (tglobal) attention
    #TODO: Missing adapter version for each model. Future question, what about a pure GCN?
    models = {'t5': 't5', 'flan-t5': 'google/flan-t5', 'long-t5-local': 'google/long-t5-local', 'long-t5-tglobal': 'google/long-t5-tglobal'}
    model_name = models[args.model]
    model_name +=  ('-' + args.model_size)
        #model_name = "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"  # self attention layer swapped with transient-global (tglobal) attention

    bool_4_args = {"no": False, "yes": True}
    length_exp_setup = {1: {"source_len": 512, "target_len": 256, "setup": "context and states"},  # 1024?
                        2: {"source_len": 512,  "target_len": 256, "setup": "only context"},
                        3: {"source_len": 256,  "target_len": 256, "setup": "only states"}}
    

    experimental_setup = args.experimental_setup
    source_len = length_exp_setup[experimental_setup]["source_len"]
    if ("google" in model_name) and (experimental_setup == 1):  # longer sequence than 2048 may not be needed...
        source_len *= 2

    target_len = length_exp_setup[experimental_setup]["target_len"]

    if 'long' not in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = LongT5ForConditionalGeneration.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, extra_ids=0, truncation=True, model_max_length=max([target_len, source_len])) 

    message_setup = length_exp_setup[experimental_setup]["setup"]
    logging.info(f"{message_setup} with...\nInput_Length: {source_len}\nOutput_Length: {target_len}")
    logger = bool_4_args[args.logger]


    model_checkpoint_name = f"baseline_{model_name}_experiment_{experimental_setup}"


    dst_metrics = DSTMetrics()  # this is loading the metrics now so we don't have to do this again



    model = load_model(model_checkpoint_name)

    accelerator = args.accelerator
    evaluate(model, tokenizer, dataloaders['test'], accelerator, 
             target_len, dst_metrics)
    if logger:
        wandb.finish()

if __name__ == '__main__':
    main()
