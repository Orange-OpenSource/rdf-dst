# ref:
# https://colab.research.google.com/drive/1-wp_pRVxl6c0Y0esn8ShIdeil3Bh854d?usp=sharing#scrollTo=WtI92WcVHrCb
# https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/text-transformers.html
# W&B integration https://docs.wandb.ai/guides/integrations/lightning
import pytorch_lightning as pl
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

    def on_test_epoch_end(self, trainer, pl_module):

        preds = pl_module.eval_epoch_outputs[0]['preds']
        labels = pl_module.eval_epoch_outputs[0]['labels']
        self.shared_evaluation(trainer, pl_module, preds, labels)

    @staticmethod
    def shared_evaluation(trainer, pl_module, preds, labels):
        rel_acc = 0.5
        pl_module.my_metrics.setdefault("relative_accuracy", rel_acc)
        pl_module.log_dict(pl_module.my_metrics, on_epoch=True)
        x = pl_module.eval_epoch_outputs['preds']
        print(x)
        print(len(x))
        
        pl_module.eval_epoch_outputs.clear()


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
