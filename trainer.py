# ref:
# https://colab.research.google.com/drive/1-wp_pRVxl6c0Y0esn8ShIdeil3Bh854d?usp=sharing#scrollTo=WtI92WcVHrCb
# https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/text-transformers.html
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

    def shared_eval_step(self, logits, labels):
        "returns preds"
        placeholder_preds = torch.full(labels.size(), -100)
        #TODO: softmax instead?
        # https://www.youtube.com/watch?v=KpKog-L9veg
        preds = torch.argmax(logits.cpu(), axis=-1)
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
        preds = self.shared_eval_step(logits, labels)
        labels = batch['labels'].cpu()
        return {"preds": preds, "labels": labels}

    def shared_eval_epoch(self, preds, labels):
        mask = labels != -100
        clean_labels = torch.masked_select(labels, mask)
        mask = preds != -100
        clean_preds = torch.masked_select(preds, mask)
        # compute metrics, masked_select already flattens the sequences.
        acc = self.acc.compute(predictions=clean_preds, references=clean_labels)
        # we have to choose an average setting because this is not binary classification
        f1 = self.f1.compute(predictions=clean_preds, references=clean_labels, average='macro')
        self.my_metrics = {'accuracy': acc['accuracy'], 'f1':f1['f1']}
        print(acc)
        print(f1)
        #TODO: Email Johanes and train it to see it isn't overfitting or underfitting
        raise SystemExit


    def validation_epoch_end(self, outputs):

        #for i, output in enumerate(outputs):
        #    print(f"batch {i}, labels {output['labels'].shape}")
        #    print(f"batch {i}, preds {output['preds'].shape}")
        #    print(f"batch {i}, loss {output['loss']}")

        preds = torch.cat([out['preds'] for out in outputs]) 
        labels = torch.cat([out['labels'] for out in outputs]) 
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        self.shared_eval_epoch(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #def on_validation_epoch_end(self):
    #TODO: try torch.cat instead
    #    all_pred_ids = torch.stack(self.val_step_outputs['preds'], 0)
    #    all_labels = torch.stack(self.val_step_outputs['labels'], 0)
    #    self.val_step_outputs.clear()
    #    self.log("accuracy", my_metrics['accuracy'], on_step=False, on_epoch=True)
    #    self.log("f1", my_metrics['f1'], on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        all_pred_ids = torch.stack(self.test_step_outputs['preds'], 0)
        all_labels = torch.stack(self.test_step_outputs['labels'], 0)
        self.test_step_outputs.clear()

#[obj for obj in lit_obj if 'val' in obj]
#['eval', 'on_predict_model_eval', 'on_test_model_eval', 'on_validation_batch_end', 'on_validation_batch_start', 'on_validation_end', 'on_validation_epoch_end', 'on_validation_epoch_start', 'on_validation_model_eval', 'on_validation_model_train', 'on_validation_start', 'val_dataloader', 'validation_epoch_end', 'validation_step', 'validation_step_end']
#[obj for obj in lit_obj if 'test' in obj]
#['on_test_batch_end', 'on_test_batch_start', 'on_test_end', 'on_test_epoch_end', 'on_test_epoch_start', 'on_test_model_eval', 'on_test_model_train', 'on_test_start', 'test_dataloader', 'test_epoch_end', 'test_step', 'test_step_end']
#[obj for obj in lit_obj if 'predict' in obj]
#['on_predict_batch_end', 'on_predict_batch_start', 'on_predict_end', 'on_predict_epoch_end', 'on_predict_epoch_start', 'on_predict_model_eval', 'on_predict_start', 'predict_dataloader', 'predict_step']

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


    #TODO: From text work in progress, see previous method
    def encode_stage(self, dialogue_hist, states_len=256, beam_search=2, repetition_penalty=2.5):
        #TODO: Tokenize dialogue history to pass input and mask to generate state method
        #encoding = ...
        self.generate_state(encoding, states_len, beam_search, repetition_penalty)



    def configure_optimizers(self):
        lr = self.lr
        optimizer = AdamW(self.parameters(), lr=lr)
        return optimizer
