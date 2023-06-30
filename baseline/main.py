from dotenv import load_dotenv
load_dotenv()  # load keys and especially w and biases to see visualizations. Looking in curr dir

import math
import os
import json
# longt5 needs special module to avoid errors
from transformers import AutoTokenizer, T5ForConditionalGeneration, LongT5ForConditionalGeneration
from utils.tools_torch import EarlyStopping, SaveBestModel
from utils.data_loader import DialogueData
from utils.args import create_arg_parser
from utils.metric_tools import DSTMetrics
from trainer import MyTrainer, MyEvaluation
from utils.predata_collate import BaselinePreDataCollator
from torch.utils.tensorboard import SummaryWriter

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes

def preprocessing(collator, dataset, num_workers, batch_size, method):

    data = DialogueData(collator, num_workers=num_workers,
                           dataset=dataset,
                           batch_size=batch_size)
    data.load_hf_data(method)
    # We tokenize in setup, but pl suggests to tokenize in prepare?
    dataloaders = data.create_loaders(subsetting=subsetting)

    train_dataloader = dataloaders["train"]
    test_dataloader = dataloaders["test"]
    validation_dataloader = dataloaders["validation"]
    return {'train': train_dataloader, 'test': test_dataloader, 'validation': validation_dataloader}


def config_train_eval(model,
                      lr,
                      weight_decay,
                      epochs,
                      num_train_optimization_steps, num_warmup_steps,
                      accelerator, version_dir):
    """
    returns trainer to use for finetuning and inference
    """


    tb_logger = SummaryWriter(version_dir)
    # flush and close happens at the end of the training loop in the other class. may not be a clean way to do this.

    total_iterations = num_train_optimization_steps


    return MyTrainer(model, tb_logger, accelerator,
                     warmup_steps=num_warmup_steps, total_steps=total_iterations,
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
    

def evaluate(model, tokenizer, test_dataloader, device, 
             target_len, dst_metrics):


    logging.info("Inference stage")

    my_evaluation = MyEvaluation(model, tokenizer, device, target_len, dst_metrics)
    my_evaluation(test_dataloader, validation=False, verbose=True)
    print(my_evaluation.results)


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

    tokenizer = AutoTokenizer.from_pretrained(model_name, extra_ids=0, truncation_side='left',
                                              truncation=True, model_max_length=max([target_len, source_len])) 

    message_setup = length_exp_setup[experimental_setup]["setup"]
    logging.info(f"{message_setup} with...\nInput_Length: {source_len}\nOutput_Length: {target_len}")
    logger = bool_4_args[args.logger]

    subsetting = bool_4_args[args.subsetting]
    store = bool_4_args[args.store_output]

    #cut_context = True if ((model_name[:2] == 't5') and (experimental_setup == 1)) else False

    dataset = args.dataset
    batch_size = args.batch
    epochs = args.epochs
    weight_decay = args.weight_decay
    lr = args.learning_rate
    num_workers = args.num_workers
    grad_acc_steps = args.gradient_accumulation_steps
    accelerator = args.accelerator
    method = args.method
    model_checkpoint_name = f"baseline_{model_name}_experiment_{experimental_setup}"

    collator = BaselinePreDataCollator(tokenizer, source_len, target_len, experimental_setup)
    logging.info("Size of the tokenizer changed in the data collator. Special tokens added, resizing token embeddings")
    model.resize_token_embeddings(len(tokenizer))
    dataloaders = preprocessing(collator, dataset, num_workers, batch_size, method)

    train_set_size = len(dataloaders['train'])
    num_train_optimization_steps = epochs * train_set_size
    num_warmup_steps = math.ceil(len(dataloaders['train']) / grad_acc_steps)

    summary = {
        "dataset": dataset,
        "max_source_length": source_len,
        "max_target_length": target_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_optimization_steps": num_train_optimization_steps,
        "num_warmup_steps": num_warmup_steps,
        "weight_decay": weight_decay,
        "learning_rate": lr,
        "training size": train_set_size
    }

    parent_dir = 'tb_logs'
    model_checkpoint_name = model_checkpoint_name.replace('google/', '')
    base_path = os.path.join(parent_dir, model_checkpoint_name)
    version_dir = create_version_num(base_path)

    checkpoint_path = os.path.join(version_dir, 'checkpoints')
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
                          accelerator,
                          version_dir)

    weights_biases_logger = {"active_logger": logger, "project": "basic_flant5", "config": summary}
    trainer.callbacks({"save": save_ckp, "early_stop": early_stopping,
                       "metrics": dst_metrics, "wandb": weights_biases_logger})


    #TODO: ADD WEIGHT AND BIASES BOOLEAN
    model_tok = training(trainer, dataloaders, tokenizer, target_len)
    model = model_tok["model"]
    tokenizer = model_tok["tokenizer"]
    results = model_tok["results"]

    # add other METRIC
    summary = dict(summary, **{"jga": results['best_epoch']['jga'],
                               "fga_exact_recall": results['best_epoch']['aga'],
                               "fuzzy_jga": results['best_epoch']['aga'],
                               "f1": results['best_epoch']['f1'],
                               "recall": results['best_epoch']['recall'],
                               "precision": results['best_epoch']['precision'],
                               "meteor": results['best_epoch']['meteor'],
                               "gleu": results['best_epoch']['gleu'],
                               "train_loss": results['best_epoch']['train_loss'],
                               "val_loss": results['best_epoch']['val_loss']
                              })

    # subprocess experiments
    # tensorboard --logdir=./baseline/tb_logs/baseline_t5-small_experiment_3/version_0
    manual_log_experiments(results, summary, checkpoint_path)

    evaluate(model, tokenizer, dataloaders['test'], accelerator, 
             target_len, dst_metrics)

if __name__ == '__main__':
    main()
