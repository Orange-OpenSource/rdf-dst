from datasets import load_dataset, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
import os
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
            self.test_dataset = self.dialogue_data['test'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['test'].column_names, batched=True) 
            if subsetting:
                self.test_dataset = self.test_dataset.select(range(10))
            return {"test": self.test_dataloader()}

        self.train_dataset = self.dialogue_data['train'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['train'].column_names, batched=True)  
        self.validation_dataset = self.dialogue_data['validation'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['validation'].column_names, batched=True) 

        if subsetting:
            self.train_dataset = self.train_dataset.select(range(25))
            self.validation_dataset = self.train_dataset.select(range(13))

        return {"train": self.train_dataloader(), "validation": self.validation_dataloader()}

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def validation_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
