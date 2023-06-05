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

        
        #from transformers import AutoTokenizer
        #tokenizer = AutoTokenizer.from_pretrained("t5-small")
        #max_input_size = set()
        #max_label_size = set()
        #count = 0

        #for t in self.train_dataset:
        #    input_ids = t["input_ids"]
        #    input_amount = torch.sum(input_ids==0)
        #    input_ids = tokenizer.decode(input_ids, skip_special_tokens=True)
        #    print(input_ids)
        #    print()
        #    max_input_size.add(input_amount)
        #    labels = t["labels"]
        #    label_amount = torch.sum(labels==-100)
        #    max_label_size.add(label_amount)
        #    labels = torch.masked_fill(labels, labels == -100, 0)
        #    labels = tokenizer.decode(labels, skip_special_tokens=True)
        #    print(labels)
        #    print()
        #    count += 1
        #    if count == 8:
        #        break
        #raise SystemExit

        # {'PMUL0322.json', 'MUL0873.json', 'MUL0143.json', 'PMUL3394.json', 'MUL1626.json', 'PMUL0396.json',
        #  'PMUL3997.json', 'MUL0767.json', 'MUL1203.json', 'PMUL2130.json', 'PMUL1729.json', 'PMUL2936.json', 'MUL1001.json',
        #  'MUL0194.json', 'PMUL2175.json', 'MUL0084.json', 'MUL0050.json',
        #  'MUL0055.json', 'MUL2630.json', 'MUL1320.json', 'MUL0711.json', 'MUL0183.json', 'PMUL2941.json',
        #  'MUL0076.json', 'PMUL1779.json', 'PMUL3627.json', 'PMUL3103.json', 'PMUL2828.json', 'MUL1697.json',
        #  'MUL1377.json', 'MUL0685.json', 'PMUL0181.json', 'MUL2174.json', 'MUL0361.json', 'PMUL4935.json', 'MUL2156.json',
        #  'PMUL4612.json', 'MUL1191.json', 'MUL0789.json', 'PMUL4499.json', 'MUL1356.json', 'MUL1522.json', 'MUL0636.json', 'PMUL1600.json', 
        # 'PMUL0271.json', 'MUL0099.json', 'PMUL2319.json', 'PMUL4906.json', 'PMUL0237.json', 'MUL1027.json', 'MUL1411.json', 'MUL1353.json', 'PMUL4885.json', 
        # 'MUL0125.json', 'MUL1221.json', 'PMUL1904.json', 'MUL1268.json', 'MUL0114.json', 'MUL0023.json', 'MUL1193.json', 'MUL2151.json', 'MUL0069.json', 
        # 'MUL0193.json', 'PMUL4031.json', 'MUL1184.json', 'PMUL2034.json', 'MUL1335.json', 'MUL1215.json', 'PMUL2590.json', 'MUL2018.json', 'PMUL2505.json', 
        # 'PMUL0460.json', 'MUL1003.json', 'MUL0047.json', 'MUL0008.json', 'PMUL2043.json', 'MUL0157.json'}

        if subsetting:
            self.train_dataset = self.train_dataset.select(range(25))
            self.test_dataset = self.train_dataset.select(range(10))
            self.validation_dataset = self.train_dataset.select(range(13))

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def validation_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
