# ref:
# https://colab.research.google.com/drive/1-wp_pRVxl6c0Y0esn8ShIdeil3Bh854d?usp=sharing#scrollTo=WtI92WcVHrCb
# https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/text-transformers.html
# W&B integration https://docs.wandb.ai/guides/integrations/lightning
import pytorch_lightning as pl
import pandas as pd
import re
import numpy as np
import torch
import evaluate
from pytorch_lightning import LightningModule
from utils.metric_tools import DSTMetrics #compute_joint_goal_accuracy
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
        dialogue_states = dict()
        i = 0
        splitRegex = re.compile(r',|\s')
        for (labels, preds) in zip(all_labels, all_preds):
            mask = labels.eq(-100)
            labels = labels.masked_fill(mask, 0)
            for l, p in zip(labels, preds):
                #TODO: Have to clean generated triplet.... very annoying!
                label_rdf = self.tokenizer.decode(l, skip_special_tokens=True)
                pred_rdf = self.tokenizer.decode(p, skip_special_tokens=True)
                #label_rdf = np.array(label_rdf.split(','))
                #pred_rdf = np.array(pred_rdf.split(','))

                label_rdf = splitRegex.split(label_rdf)
                pred_rdf = splitRegex.split(pred_rdf)
                # Removing whitespaces
                label_rdf = list(filter(lambda x: x, label_rdf))
                pred_rdf = list(filter(lambda x: x, pred_rdf))

                # TODO: Could add penalty for hallucinations!
                # reducing size of predictions to be same of labels
                pred_rdf = pred_rdf[:len(label_rdf)]

                size_label_split = len(label_rdf)/3  # triplet
                size_pred_split = len(pred_rdf)/3  # triplet

                label_rdfs = [triplet.tolist()for triplet in np.array_split(label_rdf, size_label_split)]
                pred_rdfs = [triplet.tolist() for triplet in np.array_split(pred_rdf, size_pred_split)]
                #TODO: Joint accuracy, et al.
                i += 1
                dialogue_states.setdefault(f"dst_{i}", {"prediction": pred_rdfs, "reference": label_rdfs})

        return dialogue_states

    def on_test_epoch_end(self, trainer, pl_module):


        dialogue_states = self.on_shared_epoch_end(pl_module)
        pl_module.eval_epoch_outputs.clear()
        df = pd.DataFrame(dialogue_states).T
        print(df)
        print(df['reference'].iloc[0])
        print(df['prediction'].iloc[0])

    def on_validation_epoch_end(self, trainer, pl_module):

        dialogue_states = self.on_shared_epoch_end(pl_module)
        pl_module.eval_epoch_outputs.clear()
        df = pd.DataFrame(dialogue_states).T
        print(df)

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
        self.eval_epoch_outputs = {"labels": [], "preds": []}

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

    def shared_eval_step(self, logits, labels, validation=True):
        """
        returns preds
        no need to ignore -100 as this logic has been removed
        from preprocessing. TODO: remove the labels argument
        """
        return torch.argmax(logits.cpu(), axis=-1)
        #preds = torch.argmax(logits.cpu(), axis=-1)
        #if not validation:
        #    return preds
        #placeholder_preds = torch.full(labels.size(), -100)
        ##TODO: softmax instead?
        ## https://www.youtube.com/watch?v=KpKog-L9veg
        #mask = torch.ne(labels, placeholder_preds)
        #return torch.where(mask, preds, placeholder_preds) 

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        # explicit logging, otherwise only logs epoch and steps by default
        labels = batch['labels'].cpu()
        preds = self.shared_eval_step(logits, labels)
        #TODO: monitor loss per step or epoch?
        return {"loss": loss, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx):
        _, logits = self.common_step(batch, batch_idx)
        labels = batch['labels'].cpu()
        preds = self.shared_eval_step(logits, labels, validation=False)
        return {"preds": preds, "labels": labels}

    def shared_epoch_end(self, preds, labels):
        # no need for -100...
        #labels = torch.masked_select(labels, labels!=-100)
        #preds = torch.masked_select(preds, preds!=-100)
        #TODO: torch.cat((preds1, preds2), 0) # maybe put all of them in a list, then cat

        self.eval_epoch_outputs['labels'].append(labels)
        self.eval_epoch_outputs['preds'].append(preds)

        labels = torch.flatten(labels)
        preds = torch.flatten(preds)
        # compute metrics, masked_select already flattens the sequences and removes the vals == -100
        joint_acc = 0 #compute_joint_goal_accuracy(preds, labels)

        acc = self.acc.compute(predictions=preds, references=labels)
        # we have to choose an average setting because this is not binary classification
        f1 = self.f1.compute(predictions=preds, references=labels, average='macro')
        self.my_metrics = {'encoded_accuracy': acc['accuracy'], 'encoded_f1':f1['f1'],
                           'joint_accuracy': joint_acc}

    def validation_epoch_end(self, outputs):

        preds = torch.cat([out['preds'] for out in outputs]) 
        labels = torch.cat([out['labels'] for out in outputs]) 
        self.shared_epoch_end(preds, labels)
        # lightning aggregates the loss automatically depending on params passed, doing it explicitly just to see.
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):

        preds = torch.cat([out['preds'] for out in outputs]) 
        labels = torch.cat([out['labels'] for out in outputs]) 
        self.shared_epoch_end(preds, labels)

    def configure_optimizers(self):
        lr = self.lr
        optimizer = AdamW(self.parameters(), lr=lr)
        return optimizer
