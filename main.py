# https://shivanandroy.com/fine-tune-t5-transformer-with-pytorch/
import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import math
from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.data_loader import DialogueRDFData
from utils.args import create_arg_parser
from trainer import RDFDialogueStateModel, MetricsCallback

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes


#model = AutoModel.from_pretrained("google/flan-t5-small")  # decoder_inputs and shift right instead of conditional generation. See documentation. Conditional generation does work with labels tho
def preprocessing(data_dir, tokenizer, num_workers, source_len, target_len, batch_size):

    data = DialogueRDFData(tokenizer=tokenizer, num_workers=num_workers,
                           data_dir=data_dir, source_len=source_len,
                           target_len=target_len, batch_size=batch_size)
    data.prepare_data()
    # We tokenize in setup, but pl suggests to tokenize in prepare?
    data.setup(subsetting=True)

    train_dataloader = data.train_dataloader()
    test_dataloader = data.test_dataloader()
    validation_dataloader = data.validation_dataloader()

    return {'train': train_dataloader, 'test': test_dataloader, 'validation': validation_dataloader}

def training_and_inference(model, epochs, tokenizer, lr, grad_acc_steps, dataloaders, target_len):

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
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename=checkpoint_name,
                                          mode="min",
                                          save_top_k=-1)


    early_stopping = EarlyStopping('val_loss')
    metrics = MetricsCallback()
    callbacks = [checkpoint_callback, early_stopping, metrics]
    
    trainer = pl.Trainer(max_epochs=epochs, callbacks=callbacks,
                         devices='auto', accelerator='cpu', enable_progress_bar=True)

    #trainer.tune  # tune before training to find lr??? Hyperparameter tuning!

    logging.info("Training stage")
    trainer.fit(pl_model, train_dataloaders=train_dataloader,
                val_dataloaders=validation_dataloader)  # ckpt_path to continue from ckpt

    #trainer.validate  # if I want to do more with validation

    logging.info("Inference stage")
    raise SystemExit
    #ckpt_path = './lightning_logs/version_22/checkpoints/' + checkpoint_callback.filename + '.ckpt'
    # only shuffled dialogues
    ckpt_path = './lightning_logs/version_25/checkpoints/' + checkpoint_callback.filename + '.ckpt'
    trainer.test(pl_model, dataloaders=test_dataloader, ckpt_path=ckpt_path, verbose=True)# ?

def main():

    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", extra_ids=0) 
    args = create_arg_parser()
    data_dir = args.data_dir
    batch_size = args.batch
    epochs = args.epochs
    source_len = args.source_length
    target_len = args.target_length
    lr = args.learning_rate
    num_workers = args.num_workers
    grad_acc_steps = args.gradient_accumulation_steps


    dataloaders = preprocessing(data_dir, tokenizer, num_workers, source_len, target_len, batch_size)
    training_and_inference(model, epochs, tokenizer, lr, grad_acc_steps, dataloaders, target_len)

if __name__ == '__main__':
    main()
# https://lukesalamone.github.io/posts/what-are-attention-masks/
# https://colab.research.google.com/drive/17CtsJtGCjp4YkykIpIoY0Kdb9nCadeFT?usp=sharing
# https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e

# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863?source=read_next_recirc---two_column_layout_sidebar------1---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------

# https://pub.towardsai.net/i-fine-tuned-gpt-2-on-110k-scientific-papers-heres-the-result-9933fe7c3c26?source=read_next_recirc---two_column_layout_sidebar------2---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------



