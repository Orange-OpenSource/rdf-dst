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
                 ):

        self.device = 'cuda' if cuda.is_available() else 'cpu'

        self.total_train_loss = []
        self.total_val_loss = []
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = copy.deepcopy(model)
        self.model.to(self.device)
        self.model.train()
        lr = 1e-3  # 1e-4 best so far?. 1e-3
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = LinearWarmUpScheduler(self.optimizer, warmup_steps, total_steps)

    #optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False)

    def train_loop(self, train_data, val_data, conv_type='2D', epochs=5, verbose=True, visual=False):

        early_stopping = EarlyStopping()
        save_ckp = SaveBestModel()


        early_stop_value = None
        for epoch in tqdm(range(epochs)):
            loss_curr_epoch = 0
            for step, batch in enumerate(train_data):

                self.optimizer.zero_grad()


                inputs = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                loss, _ = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss.backward()
                self.optimizer.step()
                
                self.scheduler.step()

                loss_curr_epoch += loss.item()

            train_loss = loss_curr_epoch / len(train_data)
            self.total_train_loss.append(train_loss)

            # VALIDATION
            val_loss = evaluation_loop(self.model, val_data, validation=True)

            self.total_val_loss.append(val_loss)
            if verbose:
                self.pretty_print(epoch=epoch, train_loss=train_loss, val_loss=val_loss, acc=acc)

            save_ckp(val_loss, epoch, self.model, self.optimizer, self.loss_fn)
            early_stopping(val_loss)

            if early_stopping.early_stop:
                early_stop_value = epoch+1
                print(f"Early stopping at epoch {early_stop_value}")

        if visual:
            if not early_stop_value:
                early_stop_value = None
            visualize(epochs, self.total_train_loss, self.total_val_loss, early_stop_value)

        save_model(epochs=epochs, model=self.model, optimizer=self.optimizer, criterion=self.loss_fn)

    def pretty_print(self, epoch, train_loss, val_loss, acc):
        print(f"Epoch {epoch+1}: train loss is {train_loss:.3f} | val loss is {val_loss:.3f} | Accuracy is {acc:.2f}%")




class MyEvaluation:

    def __init__(self, model, tokenizer, eval_data, validation=False):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.eval_data = eval_data
        self.validation = validation

    def evaluation_loop(verbose=False):
    
        device = 'cuda' if cuda.is_available() else 'cpu'
    
        total_loss = float('inf')
        outputs = []
        with torch.no_grad():
            for step, batch in enumerate(self.eval_data):
    
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
    
                loss, _ = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item()
    
                if not validation:
                    step_output = generate_rdfs(self.model, batch)
                    outputs.append(step_output)
                elif validation and step == len(self.eval_data) - 1:  # just generate at the end of steps before epoch
                    step_output = generate_rdfs(self.model, batch)
                    outputs.append(step_output)
                    
    
        total_loss /= len(self.eval_data)
        results = evaluate_outputs(outputs)
        if not self.validation:
            self.store_outputs(outputs)
        outputs.clear()

        ## LOG THE VAL LOSS
        #if verbose:
        #    self.pretty_print(epoch=epoch, train_loss=train_loss, val_loss=val_loss, acc=acc)
        #return total_loss

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
    


def generate_rdfs(model, batch):
    gen_kwargs = {"max_length": target_length,
                  "min_length": target_length,
                  "early_stopping": True
                 }

    generated_tokens = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs)
    decoded_preds = tokenizer.batch_decode(generated_tokens.detach(), skip_special_tokens=True)

    decoded_inputs = tokenizer.batch_decode(batch['inputs'].detach(), skip_special_tokens=True)
    labels = batch["labels"].detach()#.cpu().numpy()
    labels = torch.where(labels != -100, labels, 0)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_labels = postprocess_rdfs(decoded_labels)
    decoded_preds = postprocess_rdfs(decoded_preds)
    if isinstance(batch["dialogue_id"], list):
        dialogue_ids = batch["dialogue_id"]
    elif torch.tensor(batch["dialogue_id"]):
        dialogue_ids = batch["dialogue_id"].detach()#.cpu().numpy()

    return {"preds": decoded_preds, "labels": decoded_labels,
            "inputs": decoded_inputs, "ids": dialogue_ids}


# OLD LOOP!!!!!!!!!!!!
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

    def trainer_loop():
        pass

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
                #"top_k": 50,
                #"top_p" :0.95
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
            #decoded_preds = self.tokenizer.batch_decode(generated_tokens.detach()#.cpu().numpy(), skip_special_tokens=True)
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

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        outputs = self.shared_eval_step(batch, batch_idx)
        # release copy of graph by releasing it
        outputs.setdefault("loss", loss.item())
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
        loss = mean([out['loss'] for out in self.eval_output_list])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.on_shared_epoch_end(self.eval_output_list, validation=True)


    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.eval_output_list = []

    def on_test_epoch_end(self) -> None:
        self.on_shared_epoch_end(self.eval_output_list)

    def on_shared_epoch_end(self, outputs, validation=False):


        dst_metrics = DSTMetrics(outputs)
        results = dst_metrics()
        if not validation:
            states_df = pd.DataFrame(outputs)
            # 
            if os.getenv('DPR_JOB'):
                path = os.path.join("/userstorage/", os.getenv('DPR_JOB'))
            else:
                path = "."
            if not os.path.exists(path):
                os.makedirs(path)
            states_df.to_csv(os.path.join(path, "nested_states.csv"), index=False)

        outputs.clear()
        self.my_metrics.update(results)
        self.log_dict(self.my_metrics, on_epoch=True, sync_dist=True)

    
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
