# https://shivanandroy.com/fine-tune-t5-transformer-with-pytorch/
# https://colab.research.google.com/github/PytorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/mnist-hello-world.ipynb
# pl --> https://www.sensiocoders.com/blog/109_pl2
from dotenv import load_dotenv
load_dotenv()  # load keys and especially w and biases to see visualizations. Looking in curr dir

import wandb
import math
import re
import os
import glob
from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.data_loader import DialogueRDFData
from utils.args import create_arg_parser
from torch_trainer import MyTrainer
from utils.predata_collate import PreDataCollator
from torch.utils.tensorboard import SummaryWriter

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes

def preprocessing(collator, dataset, num_workers, batch_size):

    data = DialogueRDFData(collator, num_workers=num_workers,
                           dataset=dataset,
                           batch_size=batch_size)
    data.prepare_data()
    # We tokenize in setup, but pl suggests to tokenize in prepare?
    data.setup(subsetting=subsetting)

    train_dataloader = data.train_dataloader()
    test_dataloader = data.test_dataloader()
    validation_dataloader = data.validation_dataloader()

    return {'train': train_dataloader, 'test': test_dataloader, 'validation': validation_dataloader}


def config_train_eval(model,
                      tokenizer,
                      lr,
                      epochs, target_len, 
                      accelerator,
                      num_train_optimization_steps, num_warmup_steps, name):
    """
    returns trainer to use for finetuning and inference
    """

    parent_dir = 'tb_logs'
    # other way to log with writer?
    # https://stackoverflow.com/questions/66945431/how-to-log-metrics-eg-validation-loss-to-tensorboard-when-using-pytorch-light
    tb_logger = SummaryWriter(parent_dir)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    total_iterations = num_train_optimization_steps
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


    base_path = os.path.join(parent_dir, name)
    version_dir = create_version_num(base_path)
    checkpoint_path = os.path.join(version_dir, 'checkpoints')
    model_name_path = 'best_dst_ckpt'
    trainer = MyTrainer

    trainer =  pl.Trainer(max_epochs=epochs, callbacks=callbacks, logger=tb_logger,
                         accumulate_grad_batches=2, devices="auto",  # torch.cuda.device_count()
                         #precision="16-mixed",  # issues with precision in PL
                         strategy="ddp", accelerator=accelerator,
                         enable_progress_bar=True)
    
    pl_model = RDFDialogueStateModel(model, tokenizer, lr, epochs, num_train_optimization_steps, num_warmup_steps, target_len, store)

    tb_logger.flush()
    tb_logger.close()
    return {'trainer': trainer, 'model': pl_model}

def create_version_num(base_path):
    dirs = [d for d in os.listdir(base_path) if d.startswith("version_")]
    if dirs:
        highest_version = max(existing_versions, key=lambda x: int(x[8:]))  # Extract the version number
        version_number = int(highest_version[8:]) + 1
    else:
        version_number = 0

    new_dir = os.path.join(base_path, f"version_{version_number:01d}")
    os.makedirs(new_dir)
    return new_dir

def training(trainer, model, dataloaders):


    logging.info("Training stage")
    trainer.fit(model, train_dataloaders=dataloaders['train'],
                val_dataloaders=dataloaders['validation'])  # ckpt_path to continue from ckpt
    
    #trainer.validate  # if I want to do more with validation

    # extract the model to save it with huggingface

    #for i, (path, _) in enumerate(trainer.checkpoint_callback.best_k_models.items()):
    #    print(path)
        #m = pl_model.load_from_checkpoint(path)  # tb_logs/base_flant5_v_beta/version_0/checkpoints/best_dst_ckpt.ckpt
        #m.transformer.save_pretrained(f'{i}th_best.pt')

def evaluate(trainer, name, model, test_dataloader):

    logging.info("Inference stage")
    file_path = f'./tb_logs/{name}/'
    ckpt_path = find_version_num(file_path)

    trainer.test(model, dataloaders=test_dataloader, ckpt_path=ckpt_path, verbose=True)# ?


def find_version_num(path):

    dirs = [d for d in os.listdir(path) if 'checkpoints' in os.listdir(path + d)]
    assert dirs, "No version has any checkpoints. Did you train the model?"
    newest_version = max(map(regex_match, dirs))
    # abs path breaks last slash so adding in ckpt
    path = os.path.abspath(f"{path}version_{newest_version}/checkpoints") + "/*.ckpt"
    checkpoints = glob.glob(path)
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
    lr = args.learning_rate
    num_workers = args.num_workers
    grad_acc_steps = args.gradient_accumulation_steps
    accelerator = args.accelerator
    model_checkpoint_name = f"{model_name}_experiment_{experimental_setup}"

    collator = PreDataCollator(tokenizer, source_len, target_len, experimental_setup, cut_context=cut_context)
    dataloaders = preprocessing(collator, dataset, num_workers, batch_size)

    num_train_optimization_steps = epochs * len(dataloaders['train'])
    num_warmup_steps = math.ceil(len(dataloaders['train']) / grad_acc_steps)

    config = config_train_eval(model,
                          tokenizer,
                          lr,
                          epochs,
                          target_len,
                          accelerator,
                          num_train_optimization_steps,
                          num_warmup_steps,
                          model_checkpoint_name)
    pl_model = config['model']
    trainer = config['trainer']

    training(trainer, pl_model, dataloaders)
    evaluate(trainer, model_checkpoint_name, pl_model, dataloaders['test'])
    if logger:
        wandb.finish()

if __name__ == '__main__':
    main()
# https://lukesalamone.github.io/posts/what-are-attention-masks/
# https://colab.research.google.com/drive/17CtsJtGCjp4YkykIpIoY0Kdb9nCadeFT?usp=sharing
# https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e

# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863?source=read_next_recirc---two_column_layout_sidebar------1---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------

# https://pub.towardsai.net/i-fine-tuned-gpt-2-on-110k-scientific-papers-heres-the-result-9933fe7c3c26?source=read_next_recirc---two_column_layout_sidebar------2---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------



