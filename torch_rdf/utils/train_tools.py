import os


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """
    #def __init__(self, best_valid_loss=float('inf')):
    def __init__(self, path, model_name_path, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.path = path
        self.model_name_path = model_name_path
        # custom eval vals to be more rigorous of what model we'll be storing
        self.best_jga = -1

    def __call__(self, model, tokenizer, epoch, results_logging, log_dict):
        # more rigorous saving method with jga as well
        curr_jga = log_dict['jga']
        current_valid_loss = log_dict['val_loss']
        if (current_valid_loss < self.best_valid_loss) and (curr_jga > self.best_jga):
            self.best_valid_loss = current_valid_loss
            self.best_jga = curr_jga
            results_logging['best_epoch'] = dict(log_dict, **{'epoch': epoch})   
            curr_model_path = self.model_name_path
            if os.getenv('DPR_JOB'):
                dpr_path = os.path.join("/userstorage/", os.getenv('DPR_JOB'))
                dpr_path = os.path.join(dpr_path, self.path)
                storage_path = os.path.join(dpr_path, curr_model_path)
            else:
                storage_path = os.path.join(self.path, curr_model_path)
            model.save_pretrained(storage_path)
            tokenizer.save_pretrained(storage_path)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    # https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    def __init__(self, patience=3, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
