from dotenv import load_dotenv
load_dotenv()  # load keys and especially w and biases to see visualizations. Looking in curr dir

import wandb
import math
import subprocess
import json
import os
# longt5 needs special module to avoid errors
from transformers import AutoTokenizer, T5ForConditionalGeneration, LongT5ForConditionalGeneration
from utils.data_loader import DialogueRDFData
# an idea, maybe?
#from baseline.utils.data_loader import DialogueRDFData
from utils.args import create_arg_parser
from utils.metric_tools import DSTMetrics
from trainer import MyTrainer, MyEvaluation
from utils.predata_collate import PreDataCollator
from torch.utils.tensorboard import SummaryWriter

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
    test_dataloader = dataloaders["test"]
    validation_dataloader = dataloaders["validation"]
    return {'train': train_dataloader, 'test': test_dataloader, 'validation': validation_dataloader}


def config_train_eval(model,
                      lr,
                      weight_decay,
                      epochs,
                      dst_metrics,
                      num_train_optimization_steps, num_warmup_steps,
                      accelerator, name):
    """
    returns trainer to use for finetuning and inference
    """

    parent_dir = 'tb_logs'
    # other way to log with writer?
    # https://stackoverflow.com/questions/66945431/how-to-log-metrics-eg-validation-loss-to-tensorboard-when-using-pytorch-light
    base_path = os.path.join(parent_dir, name)
    version_dir = create_version_num(base_path)
    checkpoint_path = os.path.join(version_dir, 'checkpoints')
    model_name_path = 'best_dst_ckpt'

    tb_logger = SummaryWriter(version_dir)
    # flush and close happens at the end of the training loop in the other class. may not be a clean way to do this.

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    total_iterations = num_train_optimization_steps

    trainer = MyTrainer(model, tb_logger, accelerator, dst_metrics, warmup_steps=num_warmup_steps,
                        total_steps=total_iterations, lr=lr, epochs=epochs,
                        weight_decay=weight_decay, accumulation_steps=2,
                        verbosity=True)


    return {'trainer': trainer, 'model_name_path': model_name_path, 'checkpoint_path': checkpoint_path}

def create_version_num(base_path):

    if os.getenv('DPR_JOB'):
        path = os.path.join("/userstorage/", os.getenv('DPR_JOB'))
        base_path = os.path.join(path, base_path)
    else:
        path = "."
        base_path = os.path.join(path, base_path)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    print(base_path)
    raise SystemExit

    dirs = [d for d in os.listdir(base_path) if d.startswith("version_")]
    if dirs:
        highest_version = max(dirs, key=lambda x: int(x[8:]))  # Extract the version number
        version_number = int(highest_version[8:]) + 1
    else:
        version_number = 0

    new_dir = os.path.join(base_path, f"version_{version_number:01d}")
    os.makedirs(new_dir)
    return new_dir

def training(trainer, dataloaders, tokenizer, target_length, model_name_path, checkpoint_path):
    """
    returns model and tokenizer

    """


    logging.info("Training stage")
    return trainer.train_loop(dataloaders['train'], dataloaders['validation'], tokenizer,
                              target_length=target_length,
                              path=checkpoint_path, model_name_path=model_name_path)
    

def evaluate(model, tokenizer, test_dataloader, device, 
             target_len, dst_metrics):


    logging.info("Inference stage")


    my_evaluation = MyEvaluation(model, tokenizer, device, target_len, dst_metrics)
    my_evaluation(test_dataloader, validation=False, verbose=True)
    print(my_evaluation.results)


def manual_log_experiments(results, summary, path):
    # Save experiment logs
    with open(os.path.join(path, 'train_log.json', 'w')) as ostr:
        json.dump(results, ostr, indent=4)
    with open(os.path.join('exp_summary.json', 'w')) as ostr:
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
    length_exp_setup = {1: {"source_len": 1024, "target_len": 1024, "setup": "context and states"},
                        2: {"source_len": 512,  "target_len": 1024, "setup": "only context"},
                        3: {"source_len": 768,  "target_len": 1024, "setup": "only states"}}
    

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

    subsetting = bool_4_args[args.subsetting]
    store = bool_4_args[args.store_output]

    cut_context = True if ((model_name[:2] == 't5') and (experimental_setup == 1)) else False

    dataset = args.dataset
    batch_size = args.batch
    epochs = args.epochs
    weight_decay = args.weight_decay
    lr = args.learning_rate
    num_workers = args.num_workers
    grad_acc_steps = args.gradient_accumulation_steps
    accelerator = args.accelerator
    method = args.method
    model_checkpoint_name = f"{model_name}_experiment_{experimental_setup}"

    collator = PreDataCollator(tokenizer, source_len, target_len, experimental_setup, cut_context=cut_context)
    dataloaders = preprocessing(collator, dataset, num_workers, batch_size, method)

    train_set_size = len(dataloaders['train'])
    num_train_optimization_steps = epochs * train_set_size
    num_warmup_steps = math.ceil(len(dataloaders['train']) / grad_acc_steps)
    dst_metrics = DSTMetrics()  # this is loading the metrics now so we don't have to do this again

    config = config_train_eval(model,
                          lr,
                          weight_decay,
                          epochs,
                          dst_metrics,
                          num_train_optimization_steps,
                          num_warmup_steps,
                          accelerator,
                          model_checkpoint_name)

    model_name_path = config['model_name_path']
    checkpoint_path = config['checkpoint_path']
    trainer = config['trainer']

    if logger:
        wandb.login()  
        wandb.tensorboard.patch(root_logdir=".tb_logs/")  # save=False?, tensorboard_x=True?
        #wandb.init(project="basic_flant5", sync_tensorboard=True)

    model_tok = training(trainer, dataloaders, tokenizer, target_len, model_name_path, checkpoint_path)
    model = model_tok["model"]
    # i mean the tok does not change but whatevs
    tokenizer = model_tok["tokenizer"]
    results = model_tok["results"]

    summary = {
        "dataset": dataset,
        "max_source_length": source_len,
        "max_target_length": target_len,
        "epochs": epochs,
        "num_optimization_steps": num_train_optimization_steps,
        "num_warmup_steps": num_warmup_steps,
        "weight_decay": weight_decay,
        "learning_rate": lr,
        "training size": train_set_size,
        "jga": results['best_epoch']['jga'],
        "f1": results['best_epoch']['f1'],
        "recall": results['best_epoch']['recall'],
        "precision": results['best_epoch']['precision'],
        "meteor": results['best_epoch']['meteor'],
        "gleu": results['best_epoch']['gleu'],
        "train_loss": results['best_epoch']['train_loss'],
        "val_loss": results['best_epoch']['val_loss'],
        'git_hash': subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    }
    manual_log_experiments(results, summary, checkpoint_path)

    evaluate(model, tokenizer, dataloaders['test'], accelerator, 
             target_len, dst_metrics)
    if logger:
        wandb.finish()

if __name__ == '__main__':
    main()
# https://lukesalamone.github.io/posts/what-are-attention-masks/
# https://colab.research.google.com/drive/17CtsJtGCjp4YkykIpIoY0Kdb9nCadeFT?usp=sharing
# https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e

# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863?source=read_next_recirc---two_column_layout_sidebar------1---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------

# https://pub.towardsai.net/i-fine-tuned-gpt-2-on-110k-scientific-papers-heres-the-result-9933fe7c3c26?source=read_next_recirc---two_column_layout_sidebar------2---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------



