from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes

class DialogueRDFData(LightningDataModule):
    """
    Preprocessor for DST prediction. Need previous system utterance, previous state, and
    user utterance to yield new state?
    """

    def __init__(self, collator, num_workers: int,
                 dataset: str, batch_size: int=8,
                 ):

        super().__init__()
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        #TODO: Review dataset and multiprocessing issues 
        # https://github.com/pytorch/pytorch/issues/8976
        self.num_workers = num_workers  # no multiprocessing for now


    def prepare_data(self):
        """
        """

        txt2rdf = load_dataset("rdfdial", self.dataset).with_format("torch")
        all_data = concatenate_datasets([txt2rdf['validation'], txt2rdf['train'], txt2rdf['test']])  # splits are weird
        train_val = all_data.train_test_split(test_size=0.2)
        test_val = train_val['test'].train_test_split(test_size=0.5)
        txt2rdf.update({'train': train_val['train'], 'validation': test_val['train'], 'test': test_val['test']})


        # shuffling dialogues
        self.txt2rdf = txt2rdf.shuffle(seed=SEED)

        
    def setup(self, subsetting=True):
        """

        """

        self.train_dataset = self.txt2rdf['train'].map(self.collator, num_proc=8, remove_columns=self.txt2rdf['train'].column_names, batched=True)  
        self.validation_dataset = self.txt2rdf['validation'].map(self.collator, num_proc=8, remove_columns=self.txt2rdf['validation'].column_names, batched=True) 
        self.test_dataset = self.txt2rdf['test'].map(self.collator, num_proc=8, remove_columns=self.txt2rdf['test'].column_names, batched=True) 

        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        max_input_size = set()
        max_label_size = set()
        long_dialogues = set()
        count = 0
        for t in self.train_dataset:
            res = t["context_size"] + t["state_size"] * 3 
            #if 510 < res < 540:  # 530 is fine but 536 is not
            if 510 < res:  # 530 is fine
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
                #print(len(t["states"]))
                #print(t["state_size"])
                print()
                #print(f"Size of context {t['context_size']}")
                print(f"'Size' of input {res}")
                print()
                long_dialogues.add(t["dialogue_id"])
                count += 1
                if count == 8:
                    break
        print("TITO")
        #print(max(len(t["states"]) for t in self.train_dataset))
        #print(long_dialogues)
        raise SystemExit
        

        if subsetting:
            og_set = self.train_dataset[0]['labels'][:50]
            self.train_dataset = self.train_dataset.select(range(50))
            self.test_dataset = self.train_dataset.select(range(20))
            self.validation_dataset = self.train_dataset.select(range(26))

            new_set = self.train_dataset[0]['labels'][:50]
            compare_tensors = torch.all(torch.eq(og_set, new_set))
            assert compare_tensors, "Subset does not correspond to original dataset"
        

        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def validation_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
