from datasets import load_from_disk, dataset_dict
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from utils.data_loader import DialogueRDFData
from trainer import RDFDialogueStateModel

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes
lr = 1e-3


#model = AutoModel.from_pretrained("google/flan-t5-small")  # decoder_inputs and shift right instead of conditional generation. See documentation. Conditional generation does work with labels tho
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
tok = AutoTokenizer.from_pretrained("google/flan-t5-small", extra_ids=0) 
san_francisco = DialogueRDFData(tokenizer=tok)
san_francisco.prepare_data()
san_francisco.setup()
train_dataloader = san_francisco.train_dataloader()
test_dataloader = san_francisco.test_dataloader()
val_dataloader = san_francisco.val_dataloader()
pl_model = RDFDialogueStateModel(model, lr)
trainer = pl.Trainer(max_epochs=2, devices='auto', accelerator='cpu')
#trainer.tune  # tune before training to find lr??? Hyperparameter tuning!
#trainer.fit(pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
#            ckpt_path='./RDF_checkpoints/')
trainer.fit(pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
# consult doc https://lightning.ai/docs/pytorch/stable/common/trainer.html
#trainer.validate  # ?
#trainer.test  # ?
raise SystemExit
for i in test_dataloader:
    print(i['input_ids'].shape)
    print(tok.decode(i['input_ids'][0]))
    print(i['labels'].shape)
    print(i['labels'])
    print(i.keys())
    #print(tok.decode(i['labels'][0]))  # skip_special_tokens ???  # does not work with -100 cuz duh
    break

# TODO: Review attention masks and if I need to mask the pad tokens with -100
# https://lukesalamone.github.io/posts/what-are-attention-masks/
# https://colab.research.google.com/drive/17CtsJtGCjp4YkykIpIoY0Kdb9nCadeFT?usp=sharing
# https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e

# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863?source=read_next_recirc---two_column_layout_sidebar------1---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------

# https://pub.towardsai.net/i-fine-tuned-gpt-2-on-110k-scientific-papers-heres-the-result-9933fe7c3c26?source=read_next_recirc---two_column_layout_sidebar------2---------------------b1b02d55_82c1_4db6_9c17_af186403e94b-------



