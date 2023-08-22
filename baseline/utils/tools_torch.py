# Copyright 2023 Orange
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Software Name : knowledge-graph-dst
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: H. Andres Gonzalez

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
        self.best_f1 = float('-inf')

    def __call__(self, model, tokenizer, epoch, results_logging, log_dict):
        # more rigorous saving method with jga as well
        current_valid_loss = log_dict['val_loss']
        if (current_valid_loss < self.best_valid_loss):
            self.best_valid_loss = current_valid_loss
            results_logging['best_epoch'] = dict(log_dict, **{'epoch': epoch})   
            curr_model_path = self.model_name_path
            #if os.getenv('DPR_JOB'):
            #    dpr_path = os.path.join("/userstorage/", os.getenv('DPR_JOB'))
            #    dpr_path = os.path.join(dpr_path, self.path)
            #    storage_path = os.path.join(dpr_path, curr_model_path)
            #else:
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
