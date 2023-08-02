import wandb
from evaluator import MyEvaluation
from utils.custom_schedulers import LinearWarmupScheduler
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from codecarbon import track_emissions

import logging

logging.basicConfig(level=logging.INFO)

SEED = 42  # for replication purposes
accelerator = Accelerator()

class MyTrainer:

    def __init__(self, 
                 model, logger, device,
                 warmup_steps, eval_steps,
                 total_steps,
                 lr=1e-3,
                 epochs: int=5,
                 weight_decay: float=0.0,
                 accumulation_steps: int=2,  # careful, this increases size of graph
                 verbosity: bool=False
                 ):

        self.writer = logger
        self.model = model
        self.epochs = epochs
        self.device = device
        # no batch norm in T5? https://discuss.pytorch.org/t/accumulating-gradients/30020/3
        self.accumulation_steps = accumulation_steps


        no_decay = ["bias", "LayerNorm.weight"]
        # using parameters instead of named_parameters
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]



        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        #self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        self.scheduler = LinearWarmupScheduler(self.optimizer, warmup_steps, total_steps)
        self.eval_steps = eval_steps

        self.verbose = verbosity
        self.disable = not verbosity

        self.config = None


    def callbacks(self, dst_callbacks):
        self.dst_metrics = dst_callbacks['metrics']
        self.save_ckp = dst_callbacks['save']
        self.early_stopping = dst_callbacks['early_stop']
        self.logger = dst_callbacks['wandb']

        if self.logger['active_logger']:
            project = self.logger["project"]
            config = self.logger["config"]
            wandb.login()  
            wandb.init(project=project, config=config)

    @track_emissions(project_name='dst-base', save_to_api=True, country_iso_code='FRA',
                     experiment_id='train_baseline-2-base-dst-full', output_file='train_base_2_basefull_emissions.csv')
    def train_loop(self, train_data, val_data, tokenizer, target_length):

        train_data, val_data, self.model, self.optimizer = accelerator.prepare(train_data, val_data, self.model, self.optimizer)

        early_stop_value = None
        results_logging = {}
        for epoch in tqdm(range(self.epochs), disable=self.disable):
            loss_curr_epoch = 0
            self.model.train()
            for step, batch in enumerate(train_data):

                inputs = batch['input_ids']#.to(self.device)
                labels = batch['labels']#.to(self.device)
                attention_mask = batch['attention_mask']#.to(self.device)

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
            val_loss = my_evaluation(val_data, eval_steps=self.eval_steps, validation=True, verbose=self.verbose)
            log_dict = my_evaluation.results
            log_dict.setdefault('train_loss', train_loss)
            log_dict.setdefault('val_loss', val_loss)

            if self.logger['active_logger']:
                wandb.log(log_dict, step=epoch)

            results_logging[f'epoch_{epoch}'] = log_dict
            for metric, value in log_dict.items():
                if 'loss' in metric:
                    name_metric = metric.split('_')
                    name_metric = name_metric[1].capitalize() + '/' + name_metric[0]
                    self.writer.add_scalar(name_metric, value, epoch)
                else:
                    self.writer.add_scalar(f"{metric}/val", value, epoch)

            self.save_ckp(self.model, tokenizer, epoch, results_logging, log_dict)
            self.early_stopping(val_loss)

            if self.early_stopping.early_stop:
                early_stop_value = epoch+1
                logging.info(f"Early stopping at epoch {early_stop_value}")

            if self.verbose:
                self.pretty_print(epoch=epoch, train_loss=train_loss, val_loss=val_loss, results=log_dict)

        self.writer.flush()
        self.writer.close()
        if self.logger['active_logger']:
            wandb.finish()

        return {"model": self.model, "tokenizer": tokenizer, "results": results_logging}

    def pretty_print(self, epoch, train_loss, val_loss, results):
        logging.info(f"Epoch {epoch+1}: train loss is {train_loss:.3f} | val loss is {val_loss:.3f}")
        #for metric, value in results.items():
        #    logging.info(f"{metric}: {value}")
