from dotenv import load_dotenv
load_dotenv()  # load keys and especially w and biases to see visualizations. Looking in curr dir

import wandb
import math
import re
import os
import glob
from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.torch_data_loader import DialogueRDFData
from utils.args import create_arg_parser
from utils.metric_tools import DSTMetrics
from torch_trainer import MyTrainer, MyEvaluation
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

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    total_iterations = num_train_optimization_steps
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


    trainer = MyTrainer(model, tb_logger, accelerator, dst_metrics, warmup_steps=num_warmup_steps,
                        total_steps=total_iterations, lr=lr, epochs=epochs, accumulation_steps=2,
                        verbosity=True)


    return {'trainer': trainer, 'model_name_path': model_name_path, 'checkpoint_path': checkpoint_path}

def create_version_num(base_path):
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
    my_evaluation(test_dataloader, validation=False, verbose=False)

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
    bool_4_args = {"no": False, "yes": True}
    # should use flanT5 for longer input! --> i think the max is 2048
    length_exp_setup = {1: {"source_len": 1024, "target_len": 768, "setup": "context and states"},  # max is 1007
                        2: {"source_len": 512,  "target_len": 768, "setup": "only context"},  # max is 495 in all exp set ups. could reduce vector
                        3: {"source_len": 768,  "target_len": 768, "setup": "only states"}}  # max is 767
    

    experimental_setup = args.experimental_setup
    source_len = length_exp_setup[experimental_setup]["source_len"]
    target_len = length_exp_setup[experimental_setup]["target_len"]
    message_setup = length_exp_setup[experimental_setup]["setup"]
    logging.info(f"{message_setup} with...\nInput_Length: {source_len}\nOutput_Length: {target_len}")
    logger = bool_4_args[args.logger]

    subsetting = bool_4_args[args.subsetting]
    store = bool_4_args[args.store_output]

    if logger:
        wandb.login()  
        wandb.tensorboard.patch(root_logdir=".tb_logs/")  # save=False?, tensorboard_x=True?
        wandb.init(project="basic_flant5")


    model_name = "t5-" + args.model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    # T5 config change?  https://huggingface.co/docs/transformers/model_doc/t5
    # 0 ids so I don't have to reshape the embedding
    tokenizer = AutoTokenizer.from_pretrained(model_name, extra_ids=0) 
    cut_context = True if ((model_name[:2] == 't5') and (experimental_setup == 1)) else False

    dataset = args.dataset
    batch_size = args.batch
    epochs = args.epochs
    weight_decay = args.weight_decay
    lr = args.learning_rate
    num_workers = args.num_workers
    grad_acc_steps = args.gradient_accumulation_steps
    accelerator = args.accelerator
    method = 'online'
    model_checkpoint_name = f"{model_name}_experiment_{experimental_setup}"

    collator = PreDataCollator(tokenizer, source_len, target_len, experimental_setup, cut_context=cut_context)
    dataloaders = preprocessing(collator, dataset, num_workers, batch_size, method)

    num_train_optimization_steps = epochs * len(dataloaders['train'])
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

    #model_tok = training(trainer, dataloaders, tokenizer, target_len, model_name_path, checkpoint_path)
    #model = model_tok["model"]
    #tokenizer = model_tok["tokenizer"]

    stored_locally = True
    if stored_locally:
        model = load_model(model_checkpoint_name)

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



