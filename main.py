# https://shivanandroy.com/fine-tune-t5-transformer-with-pytorch/
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger  # tensorboard is installed with lightning, must install wandb manually
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import math
from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.data_loader import DialogueRDFData
from utils.args import create_arg_parser
from trainer import RDFDialogueStateModel, MetricsCallback, MyTrainer
from utils.predata_collate import PreDataCollator

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes

def preprocessing(tokenizer, collator, dataset, num_workers, batch_size):

    data = DialogueRDFData(collator, num_workers=num_workers,
                           dataset=dataset,
                           batch_size=batch_size)
    data.prepare_data()
    # We tokenize in setup, but pl suggests to tokenize in prepare?
    data.setup(tokenizer, subsetting=True)

    train_dataloader = data.train_dataloader()
    test_dataloader = data.test_dataloader()
    validation_dataloader = data.validation_dataloader()

    return {'train': train_dataloader, 'test': test_dataloader, 'validation': validation_dataloader}

def training_and_inference(model, epochs, tokenizer, lr, grad_acc_steps, dataloaders, target_len, store):


    name = "base_flant5_v_beta"
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
    pl_model = RDFDialogueStateModel(model, tokenizer, lr, epochs, num_train_optimization_steps, num_warmup_steps, target_len)
    # saving every time val_loss improves
    # custom save checkpoints callback pytorch lightning
    # https://github.com/Lightning-AI/lightning/issues/3096 --> to save from pretrained?
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename=checkpoint_name,
                                          mode="min",
                                          save_top_k=-1)

    early_stopping = EarlyStopping('val_loss')
    metrics = MetricsCallback()
    callbacks = [checkpoint_callback, early_stopping, metrics]
    
    trainer = MyTrainer(max_epochs=epochs, callbacks=callbacks, logger=tb_logger,
                        devices='auto', accelerator='cpu', enable_progress_bar=True)
    trainer.store = store

    #trainer.tune  # tune before training to find lr??? Hyperparameter tuning!

    logging.info("Training stage")
    trainer.fit(pl_model, train_dataloaders=train_dataloader,
                val_dataloaders=validation_dataloader)  # ckpt_path to continue from ckpt

    #trainer.validate  # if I want to do more with validation

    logging.info("Inference stage")
    ckpt_path = f'./tb_logs/{name}/version_0/checkpoints/' + checkpoint_callback.filename + '.ckpt'
    trainer.test(pl_model, dataloaders=test_dataloader, ckpt_path=ckpt_path, verbose=True)# ?

def main():

    args = create_arg_parser()
    bool_4_args = {"no": False, "yes": True}
    logger = bool_4_args[args.logger]
    if logger:
        wandb.login()  
        wandb.tensorboard.patch(root_logdir=".tb_logs/")
        wandb.init(project="basic_flant5")

    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    # 0 ids so I don't have to reshape the embedding
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", extra_ids=0) 

    store = bool_4_args[args.store_output]
    dataset = args.dataset
    experimental_setup = args.experimental_setup
    batch_size = args.batch
    epochs = args.epochs
    source_len = args.source_length
    target_len = args.target_length
    lr = args.learning_rate
    num_workers = args.num_workers
    grad_acc_steps = args.gradient_accumulation_steps

    collator = PreDataCollator(tokenizer, source_len, target_len, experimental_setup)
    dataloaders = preprocessing(tokenizer, collator, dataset, num_workers, batch_size)
    training_and_inference(model, epochs, tokenizer, lr, grad_acc_steps, dataloaders, target_len, store)
    if logger:
        wandb.finish()

if __name__ == '__main__':
    main()
# https://lukesalamone.github.io/posts/what-are-attention-masks/
# https://colab.research.google.com/drive/17CtsJtGCjp4YkykIpIoY0Kdb9nCadeFT?usp=sharing
# https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e

# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863?source=read_next_recirc---two_column_layout_sidebar------1---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------

# https://pub.towardsai.net/i-fine-tuned-gpt-2-on-110k-scientific-papers-heres-the-result-9933fe7c3c26?source=read_next_recirc---two_column_layout_sidebar------2---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------



