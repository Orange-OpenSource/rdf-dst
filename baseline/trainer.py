import pandas as pd
import torch
import os
from utils.metric_tools import DSTMetrics
from utils.postprocessing import postprocess_states
from utils.custom_schedulers import LinearWarmupScheduler
from torch.optim import AdamW
from tqdm import tqdm
from utils.tools_torch import EarlyStopping, SaveBestModel

import logging

logging.basicConfig(level=logging.INFO)

SEED = 42  # for replication purposes

class MyTrainer:

    def __init__(self, 
                 model, logger, accelerator,
                 dst_metrics,
                 warmup_steps, total_steps,
                 lr=1e-6,
                 epochs: int=5,
                 accumulation_steps: int=2,  # careful, this increases size of graph
                 verbosity: bool=False
                 ):

        self.writer = logger
        self.model = model
        self.epochs = epochs
        self.device = accelerator
        # no batch norm in T5? https://discuss.pytorch.org/t/accumulating-gradients/30020/3
        self.accumulation_steps = accumulation_steps
        lr = 1e-3  # 1e-4 best so far?. 1e-3
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = LinearWarmupScheduler(self.optimizer, warmup_steps, total_steps)

        self.dst_metrics = dst_metrics
        self.verbose = verbosity
        self.disable = not verbosity

    #optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False)

    def train_loop(self, train_data, val_data, tokenizer, target_length, path, model_name_path):

        early_stopping = EarlyStopping()
        save_ckp = SaveBestModel(path, model_name_path)

        self.model.to(self.device)

        early_stop_value = None
        for epoch in tqdm(range(self.epochs), disable=self.disable):
            loss_curr_epoch = 0
            self.model.train()
            for step, batch in enumerate(train_data):

                inputs = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                model_outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                # loss, logits, encoder_last_hidden_state, past_key_values
                loss = model_outputs.loss
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

            # VALIDATION
            my_evaluation = MyEvaluation(self.model, tokenizer, self.device, target_length, self.dst_metrics)
            val_loss = my_evaluation(val_data, validation=True, verbose=self.verbose)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            for metric, value in my_evaluation.results.items():
                self.writer.add_scalar(f"{metric}/val", value, epoch)

            save_ckp(val_loss, epoch, self.model)
            early_stopping(val_loss)

            if early_stopping.early_stop:
                early_stop_value = epoch+1
                print(f"Early stopping at epoch {early_stop_value}")

            if self.verbose:
                self.pretty_print(epoch=epoch, train_loss=train_loss, val_loss=val_loss, results=my_evaluation.results)

        self.writer.flush()
        self.writer.close()

        # don't need tokenizer during inference so not saving it
        #tokenizer.save("web_dial_dst_en_tokenizer.json")  # or is it save_pretrained? or does save_pretrained already saves the tokenizer?
        return {"model": self.model, "tokenizer": tokenizer}

    def pretty_print(self, epoch, train_loss, val_loss, results):
        print(f"Epoch {epoch+1}: train loss is {train_loss:.3f} | val loss is {val_loss:.3f}")
        for metric, value in results.items():
            print(f"{metric}: {value}")


class MyEvaluation:

    def __init__(self, model, tokenizer, device, target_length, dst_metrics):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dst_metrics = dst_metrics
        self.gen_kwargs = {"max_length": target_length,
                           "min_length": target_length,
                           "early_stopping": True
                          }
    

    def __call__(self, eval_data, validation=False, verbose=False):
    
        self.model.eval()
    
        total_loss = 0
        outputs = []
        disable = not verbose
        with torch.no_grad():
            for step, batch in tqdm(enumerate(eval_data), disable=disable):
    
                inputs = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
    
                model_outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                # loss, logits, encoder_last_hidden_state, past_key_values
                loss = model_outputs.loss
                total_loss += loss.item()
    
                if not validation:
                    step_output = self.generate_states(batch)
                    outputs.append(step_output)
                elif validation and step == len(eval_data) - 1:  # just generate at the end of steps before epoch
                    step_output = self.generate_states(batch)
                    outputs.append(step_output)

                # slower implementation
                #step_output = self.generate_states(batch)
                #outputs.append(step_output)
                    
    
        total_loss /= len(eval_data)
        self.results = self.evaluate_outputs(outputs)
        if not validation:
            self.store_outputs(outputs)
        outputs.clear()

        return total_loss

    def evaluate_outputs(self, outputs):
        return self.dst_metrics(outputs)

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
    


    def generate_states(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch["labels"].to(self.device).detach()#.cpu().numpy()

        generated_tokens = self.model.generate(input_ids, attention_mask=attention_mask, **self.gen_kwargs)
        decoded_preds = self.tokenizer.batch_decode(generated_tokens.detach(), skip_special_tokens=True)
    
        decoded_inputs = self.tokenizer.batch_decode(input_ids.detach(), skip_special_tokens=True)
        labels = torch.where(labels != -100, labels, 0)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_labels = postprocess_states(decoded_labels)
        decoded_preds = postprocess_states(decoded_preds)
        if isinstance(batch["dialogue_id"], list):
            dialogue_ids = batch["dialogue_id"]
        elif torch.tensor(batch["dialogue_id"]):
            dialogue_ids = batch["dialogue_id"].detach()#.cpu().numpy()
    
        return {"preds": decoded_preds, "labels": decoded_labels,
                "inputs": decoded_inputs, "ids": dialogue_ids}

