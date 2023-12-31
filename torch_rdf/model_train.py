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

import math
import json
import torch
import os
# longt5 needs special module to avoid errors
from transformers import AutoTokenizer, T5ForConditionalGeneration, LongT5ForConditionalGeneration
from utils.data_loader import DialogueRDFData
# an idea, maybe?
#from baseline.utils.data_loader import DialogueRDFData
from utils.train_tools import EarlyStopping, SaveBestModel
from utils.args import create_arg_parser
from utils.metric_tools import DSTMetrics
from trainer import MyTrainer
from utils.predata_collate import PreDataCollator
from torch.utils.tensorboard import SummaryWriter
from peft import get_peft_model, LoraConfig, TaskType, PrefixTuningConfig, IA3Config, AdaLoraConfig, PeftConfig, PeftModel
# PeftModel added in case of training peft with checkpoints

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes

def preprocessing(collator, dataset, num_workers, batch_size, method):

    data = DialogueRDFData(collator, num_workers=num_workers,
                           dataset=dataset,
                           batch_size=batch_size)
    data.load_hf_data(method)
    # We tokenize in setup, but pl suggests to tokenize in prepare?
    dataloaders = data.create_loaders(subsetting=subsetting)

    train_dataloader = dataloaders["train"]
    validation_dataloader = dataloaders["validation"]
    return {'train': train_dataloader, 'validation': validation_dataloader}


def config_train_eval(model,
                      lr,
                      weight_decay,
                      epochs,
                      num_train_optimization_steps, num_warmup_steps,
                      num_eval_steps,
                      device, version_dir
                      ):
    """
    returns trainer to use for finetuning and inference
    """


    tb_logger = SummaryWriter(version_dir)
    # flush and close happens at the end of the training loop in the other class. may not be a clean way to do this.

    total_iterations = num_train_optimization_steps


    return MyTrainer(model, tb_logger, device,
                     warmup_steps=num_warmup_steps, eval_steps=num_eval_steps,
                     total_steps=total_iterations,
                     lr=lr, epochs=epochs, weight_decay=weight_decay, accumulation_steps=2,
                     verbosity=True)



def create_version_num(base_path):

    if os.getenv('DPR_JOB'):
        path = os.path.join("/userstorage/", os.getenv('DPR_JOB'))
        base_path = os.path.join(path, base_path)
    else:
        path = "."
        base_path = os.path.join(path, base_path)

    if not os.path.exists(base_path):
        os.makedirs(base_path)


    dirs = [d for d in os.listdir(base_path) if d.startswith("version_")]
    if dirs:
        highest_version = max(dirs, key=lambda x: int(x[8:]))  # Extract the version number
        version_number = int(highest_version[8:]) + 1
    else:
        version_number = 0

    new_dir = os.path.join(base_path, f"version_{version_number:01d}")
    os.makedirs(new_dir)
    return new_dir

def training(trainer, dataloaders, tokenizer, target_length):
    """
    returns model and tokenizer

    """


    logging.info("Training stage")
    return trainer.train_loop(dataloaders['train'], dataloaders['validation'], tokenizer,
                              target_length=target_length)
    


def manual_log_experiments(results, summary, path):
    # Save experiment logs
    with open(os.path.join(path, 'train_log.json'), 'w') as ostr:
        json.dump(results, ostr, indent=4)
    with open(os.path.join(path, 'exp_summary.json'), 'w') as ostr:
        json.dump(summary, ostr, indent=4)


def main():

    global subsetting
    global store

    args = create_arg_parser()
    # self attention layer swapped with local attention or transient-global (tglobal) attention
    #TODO: Missing adapter version for each model. Future question, what about a pure GCN?
    models = {'t5': 't5', 'flan-t5': 'google/flan-t5', 'long-t5-local': 'google/long-t5-local', 'long-t5-tglobal': 'google/long-t5-tglobal'}
    model_name = models[args.model]
    model_path =  model_name + '-' + args.model_size

    bool_4_args = {"no": False, "yes": True}
    ignore_inter = bool_4_args[args.ig]
    length_exp_setup = {1: {"source_len": 1024, "target_len": 1024, "setup": "user, context and states"},
                        2: {"source_len": 512,  "target_len": 1024, "setup": "user and context"},
                        3: {"source_len": 1024, "target_len": 1024, "setup": "user, prev. sys input and states"},
                        4: {"source_len": 1024,  "target_len": 1024, "setup": "user and states"},
                        5: {"source_len": 128,  "target_len": 1024, "setup": "user input"},
                        6: {"source_len": 1024, "target_len": 1024, "setup": "only states"}
                        }
    
    if args.experimental_setup in [4, 5, 6]:
        abl_num = args.experimental_setup - 3
        logging.warning(f"YOU ARE RUNNING ABLATION NUMBER {abl_num}")

    experimental_setup = args.experimental_setup
    source_len = length_exp_setup[experimental_setup]["source_len"]
    peft_type = args.peft
    #if (model_name != 't5') and (experimental_setup == 1):  # longer sequence than 2048 may not be needed...
    #    source_len *= 2

    target_len = length_exp_setup[experimental_setup]["target_len"]

    dataset = args.dataset
    batch_size = args.batch
    epochs = args.epochs
    weight_decay = args.weight_decay
    lr = args.learning_rate
    num_workers = args.num_workers
    grad_acc_steps = args.gradient_accumulation_steps
    device = args.device
    method = args.method

    #peft_model_id = model_path
    #config = PeftConfig.from_pretrained(peft_model_id)

    if 'long' not in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        #model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
        #model = PeftModel.from_pretrained(model, peft_model_id)
        target_modules = None
    else:
        model = LongT5ForConditionalGeneration.from_pretrained(model_path)
        target_modules = ['q', 'v']

    tokenizer = AutoTokenizer.from_pretrained(model_path, extra_ids=0, truncation=True, truncation_side='left',
                                              model_max_length=max([target_len, source_len])) 
    #tokenizer = AutoTokenizer.from_pretrained(model_path) 
    #tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    if peft_type:
        og_model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

        peft_methods = {
            'lora': LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=4, lora_alpha=32, lora_dropout=0.1,
                               target_modules=target_modules),
            'prefix': PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20),
            'ia3': IA3Config(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, feedforward_modules=[]),
            'adalora': AdaLoraConfig(
                   init_r=12,
                   target_r=8,
                   beta1=0.85,
                   beta2=0.85,
                   tinit=200,
                   tfinal=1000,
                   deltaT=10,
                   lora_alpha=32,
                   lora_dropout=0.1,
                   task_type=TaskType.SEQ_2_SEQ_LM,
                   inference_mode=False
            )
         }

        peft_config = peft_methods[peft_type]
        model = get_peft_model(model, peft_config)
        peft_model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.print_trainable_parameters()
        logging.info(f"trainable params: {peft_model_size} || all params: {og_model_size} || trainable: {peft_model_size/og_model_size * 100}")
        logging.info(f"PEFT: {peft_config.peft_type}")
    else:
        peft_type = ''

    model_checkpoint_name = f"{model_name}_{args.model_size}_experiment_{experimental_setup}"


    message_setup = length_exp_setup[experimental_setup]["setup"]
    logging.info(f"{message_setup} with...\nInput_Length: {source_len}\nOutput_Length: {target_len}")
    logger = bool_4_args[args.logger]

    subsetting = bool_4_args[args.subsetting]
    store = bool_4_args[args.store_output]

    cut_context = True if experimental_setup == 1 else False
    #cut_context = True if ((model_name[:2] == 't5') and (experimental_setup == 1)) else False


    collator = PreDataCollator(tokenizer, source_len, target_len, experimental_setup, dataset_type=dataset,
                               ignore_inter=ignore_inter, cut_context=cut_context)
    logging.info("Size of the tokenizer changed in the data collator. Special tokens added, resizing token embeddings")
    model.resize_token_embeddings(len(tokenizer))

    device_count = torch.cuda.device_count()

    logging.info(f"There are {device_count} devices available")
    device_count = range(torch.cuda.device_count())
    
    dataloaders = preprocessing(collator, dataset, num_workers, batch_size, method)

    train_set_size = len(dataloaders['train'])
    validation_set_size = len(dataloaders['validation'])
    num_train_optimization_steps = epochs * train_set_size
    num_warmup_steps = math.ceil(train_set_size / grad_acc_steps)
    total_val_steps = validation_set_size // batch_size * epochs
    total_train_steps =  num_train_optimization_steps // batch_size

    factor = 10
    num_eval_steps = total_val_steps // factor
    while num_eval_steps == 0 and factor > 0:
        num_eval_steps = total_val_steps // factor
        factor -= 1
    if num_eval_steps == 0:
        num_eval_steps += 1
    elif num_eval_steps >= validation_set_size:
        num_eval_steps //= 2 

    summary = {
        "dataset": dataset,
        "max_source_length": source_len,
        "max_target_length": target_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "total_val_steps": total_val_steps,
        "total_train_steps": total_train_steps,
        "num_optimization_steps": num_train_optimization_steps,
        "num_warmup_steps": num_warmup_steps,
        "num_eval_steps": num_eval_steps,
        "weight_decay": weight_decay,
        "learning_rate": lr,
        "training size": train_set_size,
        "peft_type": peft_type
    }


    parent_dir = 'tb_logs'
    model_checkpoint_name = model_checkpoint_name.replace('google/', '')
    base_path = os.path.join(parent_dir, model_checkpoint_name)

    version_dir = create_version_num(base_path)

    checkpoint_path = os.path.join(version_dir, 'checkpoints', peft_type)
    model_name_path = 'best_dst_ckpt'

    dst_metrics = DSTMetrics()  # this is loading the metrics now so we don't have to do this again
    early_stopping = EarlyStopping()
    save_ckp = SaveBestModel(checkpoint_path, model_name_path)

    trainer = config_train_eval(model,
                          lr,
                          weight_decay,
                          epochs,
                          num_train_optimization_steps,
                          num_warmup_steps,
                          num_eval_steps,
                          device,
                          version_dir
                          )

    weights_biases_logger = {"active_logger": logger, "project": "basic_flant5", "config": summary}
    trainer.callbacks({"save": save_ckp, "early_stop": early_stopping,
                       "metrics": dst_metrics, "wandb": weights_biases_logger})

    model_tok = training(trainer, dataloaders, tokenizer, target_len)
    model = model_tok["model"]
    tokenizer = model_tok["tokenizer"]
    results = model_tok["results"]

    # remove validation metrics during training to speed up training
    summary = dict(summary, **{#"jga": results['best_epoch']['jga'],
                               #"fga_exact_recall": results['best_epoch']['fga_exact_recall'],
                               #"fuzzy_jga": results['best_epoch']['fuzzy_jga'],
                               #"f1": results['best_epoch']['f1'],
                               #"recall": results['best_epoch']['recall'],
                               #"precision": results['best_epoch']['precision'],
                               #"meteor": results['best_epoch']['meteor'],
                               #"gleu": results['best_epoch']['gleu'],
                               "train_loss": results['best_epoch']['train_loss'],
                               "val_loss": results['best_epoch']['val_loss']
                              })

    manual_log_experiments(results, summary, checkpoint_path)

    print(summary)
    logging.info(summary)
    logging.info(checkpoint_path)

if __name__ == '__main__':
    main()
