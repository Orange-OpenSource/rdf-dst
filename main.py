# https://shivanandroy.com/fine-tune-t5-transformer-with-pytorch/
from lightning.pytorch.loggers import TensorBoardLogger  # tensorboard is installed with lightning, must install wandb manually
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import lightning.pytorch as pl
import wandb
import math
from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.data_loader import DialogueRDFData
from utils.args import create_arg_parser
from trainer import RDFDialogueStateModel
from utils.predata_collate import PreDataCollator
from dotenv import load_dotenv

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes
load_dotenv()  # load keys and especially w and biases to see visualizations. Looking in curr dir

def preprocessing(collator, data_dir, num_workers, batch_size):

    data = DialogueRDFData(collator, num_workers=num_workers,
                           data_dir=data_dir,
                           batch_size=batch_size)
    data.prepare_data()
    # We tokenize in setup, but pl suggests to tokenize in prepare?
    data.setup()

    train_dataloader = data.train_dataloader()
    test_dataloader = data.test_dataloader()
    validation_dataloader = data.validation_dataloader()

    return {'train': train_dataloader, 'test': test_dataloader, 'validation': validation_dataloader}

def training_and_inference(model, epochs, tokenizer, lr, grad_acc_steps, dataloaders, target_len, store, name):


    tb_logger = TensorBoardLogger("tb_logs", name=name) 
    train_dataloader = dataloaders['train']
    test_dataloader = dataloaders['test']
    validation_dataloader = dataloaders['validation']

    # changing name for a more reusable version to resume training and test
    #checkpoint_name = '{epoch:02d}-{val_loss:.2f}-{encoded_accuracy:.2f}'
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
    checkpoint_name = 'best_dst_ckpt'
    num_train_optimization_steps = epochs * len(train_dataloader)
    num_warmup_steps = math.ceil(len(train_dataloader) / grad_acc_steps)
    pl_model = RDFDialogueStateModel(model, tokenizer, lr, epochs, num_train_optimization_steps, num_warmup_steps, target_len, store)
    # saving every time val_loss improves
    # custom save checkpoints callback pytorch lightning
    # https://github.com/Lightning-AI/lightning/issues/3096 --> to save from pretrained?
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename=checkpoint_name,
                                          mode="min",
                                          save_top_k=-1)

    early_stopping = EarlyStopping('val_loss', patience=3, min_delta=0)
    callbacks = [checkpoint_callback, early_stopping]
    
    trainer = pl.Trainer(max_epochs=epochs, callbacks=callbacks, logger=tb_logger,
                        devices=-1, accelerator='gpu', strategy="ddp", enable_progress_bar=True)

    #trainer.tune  # tune before training to find lr??? Hyperparameter tuning!

    logging.info("Training stage")
    trainer.fit(pl_model, train_dataloaders=train_dataloader,
                val_dataloaders=validation_dataloader)  # ckpt_path to continue from ckpt

    #trainer.validate  # if I want to do more with validation

    logging.info("Inference stage")
    ckpt_path = f'./tb_logs/{name}/version_0/checkpoints/' + checkpoint_callback.filename + '.ckpt'
    trainer.test(pl_model, dataloaders=test_dataloader, ckpt_path=ckpt_path, verbose=True)# ?

    # extract the model to save it with huggingface


    #for i, (path, _) in enumerate(trainer.checkpoint_callback.best_k_models.items()):
    #    print(path)
        #m = pl_model.load_from_checkpoint(path)  # tb_logs/base_flant5_v_beta/version_0/checkpoints/best_dst_ckpt.ckpt
        #m.transformer.save_pretrained(f'{i}th_best.pt')

def main():

    args = create_arg_parser()
    bool_4_args = {"no": False, "yes": True}
    logger = bool_4_args[args.logger]
    if logger:
        wandb.login()  
        wandb.tensorboard.patch(root_logdir=".tb_logs/")
        wandb.init(project="basic_flant5")

    #model_name = "google/flan-t5-small"
    model_name = "google/flan-t5-" + args.model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    # 0 ids so I don't have to reshape the embedding
    tokenizer = AutoTokenizer.from_pretrained(model_name, extra_ids=0) 

    store = bool_4_args[args.store_output]
    data_dir = args.data_dir
    experimental_setup = args.experimental_setup
    batch_size = args.batch
    epochs = args.epochs
    source_len = args.source_length
    target_len = args.target_length
    lr = args.learning_rate
    num_workers = args.num_workers
    grad_acc_steps = args.gradient_accumulation_steps
    model_checkpoint_name = f"flant5_{args.model}_experiment_{experimental_setup}"

    collator = PreDataCollator(tokenizer, source_len, target_len, experimental_setup)
    dataloaders = preprocessing(collator, data_dir, num_workers, batch_size)
    training_and_inference(model, epochs, tokenizer, lr, grad_acc_steps, dataloaders, target_len, store, model_checkpoint_name)
    if logger:
        wandb.finish()

if __name__ == '__main__':
    main()
# https://lukesalamone.github.io/posts/what-are-attention-masks/
# https://colab.research.google.com/drive/17CtsJtGCjp4YkykIpIoY0Kdb9nCadeFT?usp=sharing
# https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e

# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863?source=read_next_recirc---two_column_layout_sidebar------1---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------

# https://pub.towardsai.net/i-fine-tuned-gpt-2-on-110k-scientific-papers-heres-the-result-9933fe7c3c26?source=read_next_recirc---two_column_layout_sidebar------2---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------



