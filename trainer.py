# ref:
# https://colab.research.google.com/drive/1-wp_pRVxl6c0Y0esn8ShIdeil3Bh854d?usp=sharing#scrollTo=WtI92WcVHrCb
# https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/text-transformers.html

# W&B integration https://docs.wandb.ai/guides/integrations/lightning
# TODO:Evaluation https://aclanthology.org/2022.acl-short.35.pdf
# exact match reading comprehension, F1 SQUAD
import pytorch_lightning as pl
import pandas as pd
import re
import numpy as np
import torch
import evaluate
from pytorch_lightning import LightningModule
from utils.metric_tools import DSTMetrics, postprocess_rdfs #compute_joint_goal_accuracy
from torch.optim import AdamW

SEED = 42  # for replication purposes


class MetricsCallback(pl.Callback):

    def __init__(self, tokenizer):
        super().__init__()
        #DSTMetrics(self.tokenizer)
        self.tokenizer = tokenizer

    def on_shared_epoch_end(self, pl_module):

        all_preds = pl_module.eval_epoch_outputs['preds']
        all_labels = pl_module.eval_epoch_outputs['labels']
        decoded_preds = self.tokenizer.batch_decode(all_preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(all_labels, skip_special_tokens=True)
        #decoded_preds = postprocess_rdfs(decoded_preds)
        decoded_labels = postprocess_rdfs(decoded_labels)
        raise SystemExit

        df = pd.DataFrame(dialogue_states).T
        print(df)
        print(df['reference'].iloc[0])
        print(df['prediction'].iloc[0])
        return dialogue_states

    def on_test_epoch_end(self, trainer, pl_module):


        dialogue_states = self.on_shared_epoch_end(pl_module)
        pl_module.eval_epoch_outputs.clear()

    def on_validation_epoch_end(self, trainer, pl_module):

        dialogue_states = self.on_shared_epoch_end(pl_module)

    @staticmethod
    def shared_evaluation(pl_module, preds, labels):
        rel_acc = 0.5
        pl_module.my_metrics.setdefault("relative_accuracy", rel_acc)
        pl_module.log_dict(pl_module.my_metrics, on_epoch=True)


class RDFDialogueStateModel(LightningModule):

    def __init__(self, model, lr):
        super().__init__()
        self.lr = lr
        self.model = model
        self.acc = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")
        self.my_metrics = dict()
        self.eval_epoch_outputs = dict()

        self.save_hyperparameters("lr")

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)

        return outputs.loss, outputs.logits


    def common_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        return self.forward(input_ids, attention_mask, labels)

    def shared_eval_step(self, batch, batch_idx, validation=True):
        """
        returns preds
        no need to ignore -100 as this logic has been removed
        from preprocessing. TODO: remove the labels argument
        """
        gen_kwargs = {
            #"max_new_tokens": 511,
            "max_length": 512,
        }
        inputs = batch["input_ids"]
        attention = batch["attention_mask"]
        labels = batch['labels'].cpu()
        ids = batch['dialogue_id']
        generated_tokens = self.model.generate(inputs, attention_mask=attention, **gen_kwargs)

        return {"gen_tokens": generated_tokens.cpu(), "labels": labels, "ids": ids}

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        # explicit logging, otherwise only logs epoch and steps by default
        outputs = self.shared_eval_step(batch, batch_idx)
        #preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        outputs.setdefault("loss", loss)
        return outputs

    def test_step(self, batch, batch_idx):
        _, logits = self.common_step(batch, batch_idx)
        return self.shared_eval_step(batch, batch_idx)

    def shared_epoch_end(self, outputs):

        preds = torch.cat([out['gen_tokens'] for out in outputs]).cpu()
        labels = torch.cat([out['labels'] for out in outputs]).cpu()
        preds = preds.numpy()
        labels = labels.numpy()
        # 2 ways, second seems more readable? idk
        #labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, 0)

        self.eval_epoch_outputs.setdefault('preds', preds)
        self.eval_epoch_outputs.setdefault('labels', labels)


        labels = labels.flatten()
        preds = preds.flatten()
        # compute metrics, masked_select already flattens the sequences and removes the vals == -100

        acc = self.acc.compute(predictions=preds, references=labels)
        # we have to choose an average setting because this is not binary classification
        f1 = self.f1.compute(predictions=preds, references=labels, average='macro')
        self.my_metrics = {'encoded_accuracy': acc['accuracy'], 'encoded_f1':f1['f1']}

    def validation_epoch_end(self, outputs):

        self.shared_epoch_end(outputs)

        # lightning aggregates the loss automatically depending on params passed, doing it explicitly just to see.
        # for early stopping purposes
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):

        self.shared_epoch_end(outputs)

    def configure_optimizers(self):
        lr = self.lr
        optimizer = AdamW(self.parameters(), lr=lr)
        return optimizer
