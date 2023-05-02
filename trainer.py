# ref:
# https://colab.research.google.com/drive/1-wp_pRVxl6c0Y0esn8ShIdeil3Bh854d?usp=sharing#scrollTo=WtI92WcVHrCb
# https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/text-transformers.html

# W&B integration https://docs.wandb.ai/guides/integrations/lightning
# TODO:Evaluation https://aclanthology.org/2022.acl-short.35.pdf
# exact match reading comprehension, F1 SQUAD
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
import evaluate
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from utils.metric_tools import postprocess_rdfs, DSTMetrics
from torch.optim import AdamW

import logging

logging.basicConfig(level=logging.INFO)

SEED = 42  # for replication purposes


class MetricsCallback(pl.Callback):

    def __init__(self):
        self.dst_metrics = DSTMetrics()

    def on_shared_epoch_end(self, pl_module):

        decoded_preds = pl_module.eval_epoch_outputs['preds']
        #pred_linearized_rdfs = [pred["linearized_rdfs"] for pred in decoded_preds]
        #pred_rdfs = [pred["clean_rdfs"] for pred in decoded_preds]
        decoded_labels = pl_module.eval_epoch_outputs['labels']
        #label_linearized_rdfs = [label["linearized_rdfs"] for label in decoded_labels]
        #label_rdfs = [label["clean_rdfs"] for label in decoded_labels]
        dialogue_ids = pl_module.eval_epoch_outputs['dialogue_id']

        

        #TODO: For preds, labels in batch then pass and compute from there? check leos code
        results = self.linear_evaluation(decoded_preds, decoded_labels)
        #results = self.linear_evaluation(pred_linearized_rdfs, label_linearized_rdfs)
        #results = self.linear_evaluation(pred_rdfs, label_rdfs)
        raise SystemExit

        # sticking dialogues together for dialogue evaluation instead  of turn evaluation
        linearized_dialogues = self.dialogue_reconstruction(dialogue_ids, pred_linearized_rdfs, label_linearized_rdfs)

        print("DEBUGGING DIALOGUE CONSTRUCTION - EPOCH END.")
        print(dialogue_ids)
        for k in linearized_dialogues.keys():
            print(f"In dialogue {k} there's: {len(linearized_dialogues[k])} turns")
        print("DEBUGGING DIALOGUE CONSTRUCTION - EPOCH END. INIT ANOTHER EPOCH SOON")

        #return results

        #rel_acc = 0.5
        #jga = results["joint_goal_accuracy"]
        #pl_module.my_metrics.setdefault("relative_accuracy", rel_acc)
        #pl_module.log_dict(pl_module.my_metrics, on_epoch=True)


        #df = pd.DataFrame(dialogue_states).T
        #print(df)
        #print(df['reference'].iloc[0])
        #print(df['prediction'].iloc[0])

        #return dialogue_states

    def on_test_epoch_end(self, trainer, pl_module):

        results = self.on_shared_epoch_end(pl_module)
        pl_module.eval_epoch_outputs.clear()

    def on_validation_epoch_end(self, trainer, pl_module):

        results = self.on_shared_epoch_end(pl_module)
        pl_module.eval_epoch_outputs.clear()

    def linear_evaluation(self, preds, labels):
        jga = self.dst_metrics.joint_goal_accuracy(preds, labels)
        return {"joint_goal_accuracy": jga}
    
    @staticmethod
    def dialogue_reconstruction(dialogue_id, pred, label):

        dialogues = dict()
        if isinstance(dialogue_id, int):
            if dialogue_id in dialogues:
                dialogues[dialogue_id].append((pred, label))
            else:
                dialogues[dialogue_id] = [(pred, label)]
        else:
            # flattening to avoid a nested loops
            dialogue_id = [d for batch in dialogue_id for d in batch]
            pred = [p for batch in pred for p in batch]
            label = [l for batch in label for l in batch]
            for diag_id, pr, lb in zip(dialogue_id, pred, label):
                if diag_id in dialogues:
                    dialogues[diag_id].append([(pr, lb)])
                else:
                    dialogues[diag_id] = [(pr, lb)]
        
        return dialogues



class RDFDialogueStateModel(LightningModule):

    def __init__(
                 self, model,
                 tokenizer, lr,
                 epochs, num_train_optimization_steps,
                 num_warmup_steps, target_length):
        super().__init__()
        self.lr = lr
        self.model = model
        self.tokenizer = tokenizer
        self.num_training_steps = num_train_optimization_steps
        self.num_warmup_steps = num_warmup_steps
        self.target_length = target_length
        self.acc = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")
        self.my_metrics = dict()
        self.eval_epoch_outputs = dict()

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

    def shared_eval_step(self, batch, batch_idx, validation=True):
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

        return {"preds": decoded_preds, "labels": decoded_labels, "ids": batch["dialogue_id"].cpu().numpy()}

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        outputs = self.shared_eval_step(batch, batch_idx)
        outputs.setdefault("loss", loss)
        return outputs

    def test_step(self, batch, batch_idx):
        _, logits = self.common_step(batch, batch_idx)
        return self.shared_eval_step(batch, batch_idx)

    def shared_epoch_end(self, outputs):

        self.eval_epoch_outputs["labels"] = [out['labels'] for out in outputs]
        self.eval_epoch_outputs["preds"] = [out['preds'] for out in outputs]
        self.eval_epoch_outputs["dialogue_id"] = [out['ids'] for out in outputs]


        #print(self.dialogues)

        # dialogue level reconstruction for separate evaluation...
        #print("DEBUGGING DIALOGUE CONSTRUCTION - EPOCH END")
        #logging.info(self.dialogues.keys())
        #print(ids)
        #print()
        #self.dialogue_reconstruction(ids, preds, labels)
        #for k in self.dialogues.keys():
        #    print(f"In dialogue {k} there's: {len(self.dialogues[k])} turns")
        #    print(len(self.dialogues[k]))
        #print("DEBUGGING DIALOGUE CONSTRUCTION - EPOCH END. INIT ANOTHER EPOCH SOON")


    def validation_epoch_end(self, outputs):

        # lightning aggregates the loss automatically depending on params passed, doing it explicitly just to see.
        # for early stopping purposes
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.shared_epoch_end(outputs)

        #x = self.dialogues.keys()
        #print(outputs["ids"])
        #print()
        #print(x)
        #raise SystemExit

    def test_epoch_end(self, outputs):

        self.shared_epoch_end(outputs)
    

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
