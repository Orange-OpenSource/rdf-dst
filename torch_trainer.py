import pandas as pd
import torch
import torch.nn as nn
import copy
import os
from utils.metric_tools import DSTMetrics
from utils.postprocessing import postprocess_rdfs
from utils.custom_schedulers import LinearWarumupScheduler
from torch.optim import AdamW
from torch import cuda
from statistics import mean
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO)

SEED = 42  # for replication purposes

class MyTrainer:

    def __init__(self, 
                 model, lr=1e-6,
                 epochs: int=5,
                 accumulation_steps: int=2,  # careful, this increases size of graph
                 logger  # expects tensorboard logger obj
                 ):

        device = 'cuda' if cuda.is_available() else 'cpu'
        self.writer = logger
        self.model.to(device)
        self.model.train()
        self.epochs = epochs
        # no batch norm in T5? https://discuss.pytorch.org/t/accumulating-gradients/30020/3
        self.accumulation_steps = accumulation_steps
        lr = 1e-3  # 1e-4 best so far?. 1e-3
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = LinearWarmUpScheduler(self.optimizer, warmup_steps, total_steps)

    #optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False)

    def train_loop(self, train_data, val_data, path, model_name_path, verbose=False):

        early_stopping = EarlyStopping()
        save_ckp = SaveBestModel(path, model_name_path)

        device = 'cuda' if cuda.is_available() else 'cpu'

        early_stop_value = None
        disable = not verbose
        for epoch in tqdm(range(self.epochs), disable=disable):
            loss_curr_epoch = 0
            for step, batch in enumerate(train_data):

                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                loss, _ = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)

                # gradient accumulation
                loss /= self.accumulation_steps
                loss.backward()

                if ((step + 1) % self.accumulation_steps == 0) or (step + 1 == len(train_data)):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                loss_curr_epoch += loss.item()

            train_loss = loss_curr_epoch / len(train_data)
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.total_train_loss.append(train_loss)

            # VALIDATION
            val_flag = True
            my_evaluation = MyEvaluation(self.tokenizer, validation=val_flag)
            val_loss = my_evaluation.loop(self.model, val_data)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            for metric, value in my_evaluation.results.items():
                self.writer.add_scalar(f"{metric}/val", value, epoch)

            save_ckp(val_loss, epoch, self.model)
            early_stopping(val_loss)

            if early_stopping.early_stop:
                early_stop_value = epoch+1
                print(f"Early stopping at epoch {early_stop_value}")

            if verbose:
                self.pretty_print(epoch=epoch, train_loss=train_loss, val_loss=val_loss, results=my_evaluation.results)



        # SAVE TOKENIZER !  need DPR_JOB method to better save this
        self.tokenizer.save("web_dial_dst_en_tokenizer.json")  # or is it save_pretrained? or does save_pretrained already saves the tokenizer?

    def pretty_print(self, epoch, train_loss, val_loss, results):
        print(f"Epoch {epoch+1}: train loss is {train_loss:.3f} | val loss is {val_loss:.3f}")
        for metric, value in results.items():
            print(f"{metric}: {value}")


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



class MyEvaluation:

    def __init__(self, tokenizer, validation=False):
        self.tokenizer = tokenizer
        self.validation = validation
        self.results = None

    def loop(model, eval_data, verbose=False):
    
        model.eval()
        device = 'cuda' if cuda.is_available() else 'cpu'
    
        total_loss = 0
        outputs = []
        with torch.no_grad():
            for step, batch in enumerate(eval_data):
    
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
    
                loss, _ = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item()
    
                if not validation:
                    step_output = generate_rdfs(model, batch)
                    outputs.append(step_output)
                elif validation and step == len(eval_data) - 1:  # just generate at the end of steps before epoch
                    step_output = generate_rdfs(model, batch)
                    outputs.append(step_output)
                    
    
        total_loss /= len(self.eval_data)
        self.results = evaluate_outputs(outputs)
        if not self.validation:
            self.store_outputs(outputs)
        outputs.clear()

        return total_loss

    @staticmethod
    def evaluate_outputs(outputs):
        dst_metrics = DSTMetrics(outputs)
        return dst_metrics()

    @staticmethod
    def store_outputs(outputs):
        states_df = pd.DataFrame(outputs)
        # 
        if os.getenv('DPR_JOB'):
            path = os.path.join("/userstorage/", os.getenv('DPR_JOB'))
        else:
            path = "."
        if not os.path.exists(path):
            os.makedirs(path)
        states_df.to_csv(os.path.join(path, "outputs.csv"), index=False)
    


    def generate_rdfs(self, model, batch):
        gen_kwargs = {"max_length": target_length,
                      "min_length": target_length,
                      "early_stopping": True
                     }
    
        generated_tokens = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs)
        decoded_preds = self.tokenizer.batch_decode(generated_tokens.detach(), skip_special_tokens=True)
    
        decoded_inputs = self.tokenizer.batch_decode(batch['inputs'].detach(), skip_special_tokens=True)
        labels = batch["labels"].detach()#.cpu().numpy()
        labels = torch.where(labels != -100, labels, 0)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        decoded_labels = postprocess_rdfs(decoded_labels)
        decoded_preds = postprocess_rdfs(decoded_preds)
        if isinstance(batch["dialogue_id"], list):
            dialogue_ids = batch["dialogue_id"]
        elif torch.tensor(batch["dialogue_id"]):
            dialogue_ids = batch["dialogue_id"].detach()#.cpu().numpy()
    
        return {"preds": decoded_preds, "labels": decoded_labels,
                "inputs": decoded_inputs, "ids": dialogue_ids}

