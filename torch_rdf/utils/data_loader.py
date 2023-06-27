from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
import torch
from torch.utils.data import DataLoader

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes

class DialogueRDFData:
    """
    Preprocessor for DST prediction. Need previous system utterance, previous state, and
    user utterance to yield new state?
    """

    def __init__(self, collator, num_workers: int,
                 dataset: str, batch_size: int=8,
                 ):

        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        #TODO: Review dataset and multiprocessing issues 
        # https://github.com/pytorch/pytorch/issues/8976
        self.num_workers = num_workers  # no multiprocessing for now


    def load_hf_data(self, method):
        """
        """
        if method == "local":
            path = self.dataset + "_rdf_data"
            dialogue_data = load_from_disk(path).with_format("torch")
            # https://huggingface.co/docs/datasets/cache
            dialogue_data.cleanup_cache_files()
        else:
            dialogue_data = load_dataset("rdfdial", self.dataset).with_format("torch")
            all_data = concatenate_datasets([dialogue_data['validation'], dialogue_data['train'], dialogue_data['test']])  # splits are weird
            train_val = all_data.train_test_split(test_size=0.2)
            test_val = train_val['test'].train_test_split(test_size=0.5)
            dialogue_data.update({'train': train_val['train'], 'validation': test_val['train'], 'test': test_val['test']})


        # shuffling dialogues
        self.dialogue_data = dialogue_data.shuffle(seed=SEED)

        
    def create_loaders(self, subsetting=True):
        """

        """

        self.train_dataset = self.dialogue_data['train'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['train'].column_names, batched=True)  
        self.validation_dataset = self.dialogue_data['validation'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['validation'].column_names, batched=True) 
        self.test_dataset = self.dialogue_data['test'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['test'].column_names, batched=True) 

        #self.debugging_lengths()

        if subsetting:
            self.train_dataset = self.train_dataset.select(range(25))
            self.test_dataset = self.train_dataset.select(range(10))
            self.validation_dataset = self.train_dataset.select(range(13))

        return {"train": self.train_dataloader(), "validation": self.validation_dataloader(), "test": self.test_dataloader()}

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def validation_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def debugging_lengths(self):

        max_input_size = set()
        max_label_size = set()
        count = 0

        tokenizer = self.collator.tokenizer

        for t in self.train_dataset:

            if (len(t["states"]) > 50) and (len(t['txt']) > 600):
                print(len(t['txt']))
                input_ids = t["input_ids"]
                input_amount = torch.sum(input_ids==0)
                input_ids = tokenizer.decode(input_ids, skip_special_tokens=True)
                print(input_ids)
                print()
                max_input_size.add(input_amount)
                labels = t["labels"]
                label_amount = torch.sum(labels==-100)
                max_label_size.add(label_amount)
                labels = torch.masked_fill(labels, labels == -100, 0)
                labels = tokenizer.decode(labels, skip_special_tokens=True)
                print(labels)
                print()
                print(t["states"])
                print()
                count += 1
                if count == 16:
                    break
        raise SystemExit
