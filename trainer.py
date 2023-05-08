# ref:
# https://colab.research.google.com/drive/1-wp_pRVxl6c0Y0esn8ShIdeil3Bh854d?usp=sharing#scrollTo=WtI92WcVHrCb
# https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/text-transformers.html

# W&B integration https://docs.wandb.ai/guides/integrations/lightning
# TODO:Evaluation https://aclanthology.org/2022.acl-short.35.pdf
# exact match reading comprehension, F1 SQUAD
import lightning.pytorch as pl
import numpy as np
import torch
import re
from transformers import get_linear_schedule_with_warmup
from utils.metric_tools import DSTMetrics
from torch.optim import AdamW

import logging

logging.basicConfig(level=logging.INFO)

SEED = 42  # for replication purposes


def postprocess_rdfs(decoded_batch):
    """
    returns several rdf triplets per batch
    """

    regexSplit = re.compile(r"(?<!\s),(?!\s)")
    decoded_batch = [regexSplit.split(row) for row in decoded_batch]
    decoded_batch = [[word.strip() for word in rdfs] for rdfs in decoded_batch]
    # casting set to list to facilitate flattening before computing metrics
    clean_rdfs = [list(set([tuple(rdfs[i:i+3]) for i in range(0, len(rdfs), 3)])) for rdfs in decoded_batch]
    return clean_rdfs

    
class RDFDialogueStateModel(pl.LightningModule):

    def __init__(
                 self, model,
                 tokenizer, lr,
                 epochs, num_train_optimization_steps,
                 num_warmup_steps, target_length,
                 store):

        super().__init__()
        self.store = store
        self.lr = lr
        self.model = model
        self.tokenizer = tokenizer
        self.num_training_steps = num_train_optimization_steps
        self.num_warmup_steps = num_warmup_steps
        self.target_length = target_length
        self.my_metrics = dict()

        self.save_hyperparameters("lr", "epochs")

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

    def shared_eval_step(self, batch, batch_idx):
        """
        returns preds
        """
        # https://huggingface.co/blog/how-to-generate
        # https://huggingface.co/docs/transformers/v4.28.1/en/generation_strategies
        with torch.no_grad():
            # https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationConfig
            gen_kwargs = {
                #"max_new_tokens": 511,
                "max_length": self.target_length,
                "min_length": self.target_length,
                "early_stopping": True
            }
    #            generated_ids = self.model.generate(input_ids=input_ids,
    #                                                attention_mask=attention_mask,
    #                                                max_length=states_len,
    #                                                truncation=True,
    #                                                num_beams=beam_search,
    #                                                repetition_penalty=repetition_penalty,
    #                                                length_penalty=1.0,
    #                                                early_stopping=True)

    #            prediction = [self.tokenizer.decode(state, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            generated_tokens = self.model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs)
            decoded_preds = self.tokenizer.batch_decode(generated_tokens.detach().cpu().numpy(), skip_special_tokens=True)
            labels = batch["labels"].detach().cpu().numpy()
            labels = np.where(labels != -100, labels, 0)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            decoded_labels = postprocess_rdfs(decoded_labels)
            decoded_preds = postprocess_rdfs(decoded_preds)

            if isinstance(batch["dialogue_id"], list):
                dialogue_ids = batch["dialogue_id"]
            elif torch.tensor(batch["dialogue_id"]):
                dialogue_ids = batch["dialogue_id"].cpu().numpy()

        return {"preds": decoded_preds, "labels": decoded_labels, "ids": dialogue_ids}

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        outputs = self.shared_eval_step(batch, batch_idx)
        outputs.setdefault("loss", loss)
        self.eval_output_list.append(outputs)

    def test_step(self, batch, batch_idx):
        _, logits = self.common_step(batch, batch_idx)
        outputs = self.shared_eval_step(batch, batch_idx)
        self.eval_output_list.append(outputs)


##https://github.com/Lightning-AI/lightning/pull/16520
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.eval_output_list = []
   
    def on_validation_epoch_end(self) -> None:
        loss = torch.stack([out['loss'] for out in self.eval_output_list]).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.on_shared_epoch_end(self.eval_output_list, validation=True)


    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.eval_output_list = []

    def on_test_epoch_end(self) -> None:
        self.on_shared_epoch_end(self.eval_output_list)

    def on_shared_epoch_end(self, outputs, validation=False):


        dst_metrics = DSTMetrics(outputs)
        if validation:
            results = dst_metrics.compute()
        elif self.store:
            results = dst_metrics.compute(store=self.store)
        outputs.clear()
        self.my_metrics.update(results)
        self.log_dict(self.my_metrics, on_epoch=True)
    
    # https://discuss.huggingface.co/t/t5-finetuning-tips/684/3

    def configure_optimizers(self):
        lr = self.lr
        optimizer = AdamW(self.parameters(), lr=lr)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                                     num_training_steps=self.num_training_steps,
                                                                     num_warmup_steps=self.num_warmup_steps),
                        'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
