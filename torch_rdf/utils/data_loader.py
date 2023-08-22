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


from datasets import load_dataset, concatenate_datasets, load_from_disk, disable_caching
from torch.utils.data import DataLoader
import os
import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes
disable_caching()  # we have a lot of different collators that depend on data sets and experimental setup. We have to avoid reloading previous maps

class DialogueRDFData:
    """
    Preprocessor for DST prediction. Need previous system utterance, previous state, and
    user utterance to yield new state?
    """

    def __init__(self, collator, num_workers: int,
                 dataset: str, batch_size: int=8,
                 inference_time: bool=False
                 ):

        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        #TODO: Review dataset and multiprocessing issues 
        # https://github.com/pytorch/pytorch/issues/8976
        self.num_workers = num_workers  # no multiprocessing for now
        self.inference_time = inference_time


    def load_hf_data(self, method):
        """
        """
        if method == "offline":
            path = self.dataset + "_rdf_data"
            path = os.path.join('..', path)
            dialogue_data = load_from_disk(path).with_format("torch")
            # https://huggingface.co/docs/datasets/cache
            dialogue_data.cleanup_cache_files()
        else:
            dialogue_data = load_dataset("rdfdial", self.dataset, download_mode='force_redownload').with_format("torch")
            all_data = concatenate_datasets([dialogue_data['validation'], dialogue_data['train'], dialogue_data['test']])  # splits are weird
            train_val = all_data.train_test_split(test_size=0.2)
            # I need to split this, otherwise the model would see the data during validation!
            test_val = train_val['test'].train_test_split(test_size=0.5)
            dialogue_data.update({'train': train_val['train'], 'validation': test_val['train'], 'test': test_val['test']})


        # shuffling dialogues
        self.dialogue_data = dialogue_data.shuffle(seed=SEED)

        
    def create_loaders(self, subsetting=True):
        """

        """

        if self.inference_time:
            self.test_dataset = self.dialogue_data['test'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['test'].column_names,
                                                               batched=True, load_from_cache_file=False) 
            if subsetting:
        
                subset_val = round(len(self.test_dataset) * .35)
                self.test_dataset = self.test_dataset.select(range(subset_val))
            return {"test": self.test_dataloader()}

        self.train_dataset = self.dialogue_data['train'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['train'].column_names,
                                                             batched=True, load_from_cache_file=False)  
        self.validation_dataset = self.dialogue_data['validation'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['validation'].column_names,
                                                                       batched=True, load_from_cache_file=False) 

        if subsetting:
            subset_val = round(len(self.train_dataset) * .35)
            self.train_dataset = self.train_dataset.select(range(subset_val))
            subset_val = round(len(self.validation_dataset) * .35)
            self.validation_dataset = self.validation_dataset.select(range(subset_val))
        
        return {"train": self.train_dataloader(), "validation": self.validation_dataloader()}

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def validation_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
