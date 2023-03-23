import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.optim import AdamW

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes


class RDFDialogueStateModel(LightningModule):

    def __init__(self, model, lr):
        super().__init__()
        self.lr = lr
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)

        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        loss, logits = self.forward(input_ids, attention_mask, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        loss, logits = self.forward(input_ids, attention_mask, labels)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        loss, logits = self.forward(input_ids, attention_mask, labels)
        return loss

    def configure_optimizers(self):
        lr = self.lr
        optimizer = AdamW(self.parameters(), lr=lr)
        return optimizer
