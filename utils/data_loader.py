from datasets import load_dataset
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from utils.predata_collate import PreDataCollator
import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes

class DialogueRDFData(LightningDataModule):
    """
    Preprocessor for DST prediction. Need previous system utterance, previous state, and
    user utterance to yield new state?
    """

    def __init__(self, tokenizer, num_workers: int,
                 source_len: int, target_len: int,
                 data_dir: str, batch_size: int=6,
                 ):

        super().__init__()
        self.source_len = source_len
        self.target_len = target_len
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        #TODO: Review dataset and multiprocessing issues 
        # https://github.com/pytorch/pytorch/issues/8976
        self.num_workers = num_workers  # no multiprocessing for now


    def prepare_data(self):
        """
        """

        data_files = {"train": self.data_dir + "train.jsonl", "test": self.data_dir + "test.jsonl", "validation": self.data_dir + "validation.jsonl"}
        #txt2rdf = load_dataset("json", data_files=data_files).with_format("torch")
        txt2rdf = load_dataset("rdfdial", "multiwoz").with_format("torch")
        print(txt2rdf)
        txt2rdf = load_dataset("rdfdial", "multiwoz").with_format("torch")
        print("TITO")
        print("\n"*7)
        print(txt2rdf)
        raise SystemExit

        # https://huggingface.co/docs/datasets/v1.12.0/cache.html cleaning cache to see changes in data collator during debugging
        txt2rdf.cleanup_cache_files()  # load_from_cache=False in map???

        # shuffling dialogues
        self.txt2rdf = txt2rdf.shuffle(seed=SEED)


        
    def setup(self, subsetting=True):
        """

        """

        collator = PreDataCollator(self.tokenizer, self.source_len, self.target_len)
        self.train_dataset = self.txt2rdf['train'].map(collator, num_proc=8, remove_columns=self.txt2rdf['train'].column_names, batched=True)  
        self.validation_dataset = self.txt2rdf['validation'].map(collator, num_proc=8, remove_columns=self.txt2rdf['validation'].column_names, batched=True) 
        self.test_dataset = self.txt2rdf['test'].map(collator, num_proc=8, remove_columns=self.txt2rdf['test'].column_names, batched=True) 


        if subsetting:
            og_set = self.train_dataset[0]['labels'][:50]
            self.train_dataset = Subset(self.train_dataset, range(100))
            self.test_dataset = Subset(self.test_dataset, range(40))
            self.validation_dataset = Subset(self.validation_dataset, range(52))

            new_set = self.train_dataset[0]['labels'][:50]
            compare_tensors = torch.all(torch.eq(og_set, new_set))
            assert compare_tensors, "Subset does not correspond to original dataset"
        

    #TODO: change workers to 12 when testing with gpu
    # shuffling turns between dialogues
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=8, num_workers=self.num_workers, shuffle=False)

    def validation_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=8, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=8, num_workers=self.num_workers, shuffle=False)

