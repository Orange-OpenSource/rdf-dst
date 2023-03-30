from datasets import load_from_disk, dataset_dict
import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import re
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from utils.data_loader import DialogueRDFData
from utils.args import create_arg_parser
from trainer import RDFDialogueStateModel, MetricsCallback

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes


#model = AutoModel.from_pretrained("google/flan-t5-small")  # decoder_inputs and shift right instead of conditional generation. See documentation. Conditional generation does work with labels tho
def preprocessing(data_dir, tokenizer, num_workers, max_len, batch_size, data_type):
    data = DialogueRDFData(tokenizer=tokenizer, num_workers=num_workers,
                           data_dir=data_dir, max_len=max_len,
                           batch_size=batch_size, data_type=data_type)
    data.prepare_data()
    data.setup(subsetting=True)

    train_dataloader = data.train_dataloader()
    test_dataloader = data.test_dataloader()
    val_dataloader = data.val_dataloader()

    return {'train': train_dataloader, 'test': test_dataloader, 'dev': val_dataloader}

def training_and_inference(model, epochs, tokenizer, lr, dataloaders):

    train_dataloader = dataloaders['train']
    test_dataloader = dataloaders['test']
    val_dataloader = dataloaders['dev']

    checkpoint_name = 'dst-epoch{epoch+1:02d}-{val_loss:.2f}'
    pl_model = RDFDialogueStateModel(model, lr)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename=checkpoint_name)  # d is pad

    early_stopping = EarlyStopping('val_loss')
    metrics = MetricsCallback(tokenizer)
    callbacks = [checkpoint_callback, early_stopping, metrics]
    
    trainer = pl.Trainer(max_epochs=epochs, callbacks=callbacks,
                         devices='auto', accelerator='cpu')
    #trainer.tune  # tune before training to find lr??? Hyperparameter tuning!

    logging.info("Training stage")
    trainer.fit(pl_model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)  # ckpt_path to continue from ckpt

    #trainer.validate  # if I want to do more with validation

    #logging.info("Inference stage")
    #ckpt_path = './lightning_logs/version_0/checkpoints/' + checkpoint_name
    #trainer.test(pl_model, ckpt_path=ckpt_path, verbose=True)# ?

def main():

    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", extra_ids=0) 
    all_data_types = {'sfx': 'SF', 'multiwoz': 'MWOZ', 'dstc2': 'DSTC'}
    args = create_arg_parser()
    data_dir = args.data_dir
    batch_size = args.batch
    epochs = args.epochs
    max_len = args.seq_length
    lr = args.learning_rate
    num_workers = args.num_workers


    dataRegex = re.compile(r"(\w+_)")
    res = dataRegex.search(data_dir)
    data_key = res.group()[:-5]  #TODO: Fix the regex above to make this less brute force
    data_type = all_data_types[data_key]

    dataloaders = preprocessing(data_dir, tokenizer, num_workers, max_len, batch_size, data_type)
    training_and_inference(model, epochs, tokenizer, lr, dataloaders)

if __name__ == '__main__':
    main()
# https://lukesalamone.github.io/posts/what-are-attention-masks/
# https://colab.research.google.com/drive/17CtsJtGCjp4YkykIpIoY0Kdb9nCadeFT?usp=sharing
# https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e

# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863?source=read_next_recirc---two_column_layout_sidebar------1---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------

# https://pub.towardsai.net/i-fine-tuned-gpt-2-on-110k-scientific-papers-heres-the-result-9933fe7c3c26?source=read_next_recirc---two_column_layout_sidebar------2---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------



