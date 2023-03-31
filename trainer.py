# ref:
# https://colab.research.google.com/drive/1-wp_pRVxl6c0Y0esn8ShIdeil3Bh854d?usp=sharing#scrollTo=WtI92WcVHrCb
# https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/text-transformers.html
# W&B integration https://docs.wandb.ai/guides/integrations/lightning
import pytorch_lightning as pl
import torch
import evaluate
from pytorch_lightning import LightningModule
from torch.optim import AdamW

SEED = 42  # for replication purposes


class MetricsCallback(pl.Callback):

    def __init__(self, tokenizer):
        super().__init__()
        self.metrics = []
        self.tokenizer = tokenizer



    #TODO: Move decoding and generation to another class where I can tokenize?
    def generate_state(self, encoding, states_len, beam_search, repetition_penalty):

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                max_length=states_len,
                                                truncation=True,
                                                num_beams=beam_search,
                                                repetition_penalty=repetition_penalty,
                                                length_penalty=1.0,
                                                early_stopping=True)

            prediction = [self.tokenizer.decode(state, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]


    # TODO: LAST STEP, THE GOAL HERE IS TO PASS SOME DIALOGUE AND GENERATE STATES
    #TODO: From text work in progress, see previous method
    def encode_stage(self, dialogue_hist, states_len=256, beam_search=2, repetition_penalty=2.5):
        #TODO: Tokenize dialogue history to pass input and mask to generate state method
        #encoding = ...
        self.generate_state(encoding, states_len, beam_search, repetition_penalty)


class RDFDialogueStateModel(LightningModule):

    def __init__(self, model, lr):
        super().__init__()
        self.lr = lr
        self.model = model
        self.acc = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")
        self.my_metrics = dict()

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
        "returns preds"
        preds = torch.argmax(logits.cpu(), axis=-1)
        if not validation:
            return preds
        placeholder_preds = torch.full(labels.size(), -100)
        #TODO: softmax instead?
        # https://www.youtube.com/watch?v=KpKog-L9veg
        mask = torch.ne(labels, placeholder_preds)
        return torch.where(mask, preds, placeholder_preds) 

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


    def validation_epoch_end(self, outputs):

        preds = torch.cat([out['preds'] for out in outputs]) 
        labels = torch.cat([out['labels'] for out in outputs]) 
        # lightning aggregates the loss automatically depending on params passed, doing it explicitly just to see.
        loss = torch.stack([out['loss'] for out in outputs]).mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        labels = torch.masked_select(labels, labels!=-100)
        preds = torch.masked_select(preds, preds!=-100)
        # compute metrics, masked_select already flattens the sequences and removes the vals == -100
        acc = self.acc.compute(predictions=preds, references=labels)
        # we have to choose an average setting because this is not binary classification
        f1 = self.f1.compute(predictions=preds, references=labels, average='macro')
        self.my_metrics = {'encoded_accuracy': acc['accuracy'], 'encoded_f1':f1['f1']}
        self.log_dict(self.my_metrics, on_epoch=True)

    def test_epoch_end(self, outputs):

        preds = torch.cat([out['preds'] for out in outputs]) 
        labels = torch.cat([out['labels'] for out in outputs]) 
        print(preds)

    def configure_optimizers(self):
        lr = self.lr
        optimizer = AdamW(self.parameters(), lr=lr)
        return optimizer
