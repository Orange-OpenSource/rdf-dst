# ref:
# https://colab.research.google.com/drive/1-wp_pRVxl6c0Y0esn8ShIdeil3Bh854d?usp=sharing#scrollTo=WtI92WcVHrCb
import pytorch_lightning as pl
import torch
import evaluate
from pytorch_lightning import LightningModule
from torch.optim import AdamW

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes


class RDFDialogueStateModel(LightningModule):

    def __init__(self, model, tokenizer, lr):
        super().__init__()
        self.lr = lr
        self.model = model
        self.tokenizer = tokenizer
        self.metric_acc = evaluate.load("accuracy")
        self.metric_f1 = evaluate.load("f1")
        self.my_metrics = dict()

        self.test_step_outputs = []

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

    def training_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        self.test_step_outputs.append(logits)
        return logits

    #def on_validation_batch_end(self, out, batch, batch_idx):
    def on_test_epoch_end(self):
        # This can be in a callback but should it? Adding functionality to gauge rdf generation
        logging.info("epoch end")

        preds = torch.argmax(logits, dim=-1)
        print(preds.shape)
        print(preds)
        x = self.tokenizer.decode(preds.tolist()[0])  # decode per prediction in batch
        print(x)
        #encoding = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        #self.generate_state(encoding)
        self.test_step_outputs.clear()  # free memory
        print("TITO")
        raise SystemExit

#[obj for obj in lit_obj if 'val' in obj]
#['eval', 'on_predict_model_eval', 'on_test_model_eval', 'on_validation_batch_end', 'on_validation_batch_start', 'on_validation_end', 'on_validation_epoch_end', 'on_validation_epoch_start', 'on_validation_model_eval', 'on_validation_model_train', 'on_validation_start', 'val_dataloader', 'validation_epoch_end', 'validation_step', 'validation_step_end']
#[obj for obj in lit_obj if 'test' in obj]
#['on_test_batch_end', 'on_test_batch_start', 'on_test_end', 'on_test_epoch_end', 'on_test_epoch_start', 'on_test_model_eval', 'on_test_model_train', 'on_test_start', 'test_dataloader', 'test_epoch_end', 'test_step', 'test_step_end']
#[obj for obj in lit_obj if 'predict' in obj]
#['on_predict_batch_end', 'on_predict_batch_start', 'on_predict_end', 'on_predict_epoch_end', 'on_predict_epoch_start', 'on_predict_model_eval', 'on_predict_start', 'predict_dataloader', 'predict_step']

    #TODO: Move decoding and generation to another class where I can tokenize?
    def generate_state(self, encoding, states_len, beam_search, repetition_penalty):

        input_ids, attention_mask = encoding['input_ids']
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



    def pl_compute_metrics(self, logits, labels):
        #TODO: move it to a callback
        logits = logits.cpu()
        labels = labels.cpu()
        #pred_ids = torch.flatten(torch.argmax(logits, axis=-1), 0)  # stack instead of flatten?
        pred_ids = torch.stack(torch.argmax(logits, axis=-1), 0)  # is it working?
        labels = torch.flatten(labels, 0)
        mask = labels != -100
        clean_labels = torch.masked_select(labels, mask)
        clean_preds = torch.masked_select(pred_ids, mask)
        acc = self.metric_acc.compute(predictions=clean_preds, references=clean_labels)
        f1 = self.metric_f1.compute(predictions=clean_preds, references=clean_labels, average='macro')
        #print({'accuracy': acc['accuracy'], 'f1':f1['f1']})
        self.my_metrics = {'accuracy': acc['accuracy'], 'f1':f1['f1']}
        

    def configure_optimizers(self):
        lr = self.lr
        optimizer = AdamW(self.parameters(), lr=lr)
        return optimizer
