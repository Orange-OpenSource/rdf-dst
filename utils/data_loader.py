from datasets import load_from_disk, dataset_dict
from sklearn.model_selection import GroupShuffleSplit
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
                 max_len: int, data_dir: str, batch_size: int=6,
                 data_type: str='SF'):
        super().__init__()
        self.max_len = max_len
        self.data_dir = data_dir
        self.data_type = data_type
        self.tokenizer = tokenizer
        #TODO: Review dataset and multiprocessing issues 
        # https://github.com/pytorch/pytorch/issues/8976
        self.num_workers = num_workers  # no multiprocessing for now


    def prepare_data(self):
        """
        SF has only a train split
        """
        rdf_dir = self.data_dir + 'rdf'
        # important to pass format so dataloader loads them as tensors and not lists
        txt2rdf = load_from_disk(rdf_dir).with_format("torch")

        # https://huggingface.co/docs/datasets/v1.12.0/cache.html cleaning cache to see changes in data collator during debugging
        txt2rdf.cleanup_cache_files()  # load_from_cache=False in map???

        collator = PreDataCollator(self.tokenizer, self.max_len)

        txt2rdf = txt2rdf.shuffle(seed=SEED)

        if len(txt2rdf) == 1:
            train_dataset = txt2rdf['train']
            #train_dataset = train_dataset.map(collator, remove_columns=self.history, num_proc=8, batched=True)
            train_dataset = train_dataset.map(collator, num_proc=8, remove_columns=train_dataset.column_names, batched=True)
            self.data_for_model = dataset_dict.DatasetDict({'train': train_dataset})


        elif len(txt2rdf) == 2:
            train_dataset = txt2rdf['train']
            test_dataset = txt2rdf['test']
            train_dataset = train_dataset.map(collator, num_proc=8, remove_columns=train_dataset.column_names, batched=True)  
            test_dataset = test_dataset.map(collator, num_proc=8, remove_columns=test_dataset.column_names, batched=True) 
            self.data_for_model = dataset_dict.DatasetDict({'train': train_dataset, 'test': test_dataset})

        elif len(txt2rdf) == 3:
            train_dataset = txt2rdf['train']
            test_dataset = txt2rdf['test']
            dev_dataset = txt2rdf['dev']

            train_dataset = train_dataset.map(collator, num_proc=8, remove_columns=train_dataset.column_names, batched=True)  
            test_dataset = test_dataset.map(collator, num_proc=8, remove_columns=test_dataset.column_names, batched=True) 
            dev_dataset = dev_dataset.map(collator, num_proc=8, remove_columns=dev_dataset.column_names, batched=True) 

            self.data_for_model = dataset_dict.DatasetDict({'train': train_dataset, 'test': test_dataset, 'dev': dev_dataset})

        else:
            raise Exception("No data splits provided")
        

    def setup(self, subsetting=True):
        """

        To make sure the data splitting process does not have turns from the same dialogue in different splits, we are splitting
        using sklearn and not hf's API. The API does not have a splitting by groups functionality
        Added subsetting option to use fewer data points for debugging purposes
        """

        size_data = len(self.data_for_model)
        if size_data == 1:
            train_dataset = self.data_for_model['train']
            train_dataset, test_dataset = self.splitter_by_groups(train_dataset, 'dialogue_id', n_splits=1, train_size=.85)
            train_dataset, dev_dataset = self.splitter_by_groups(train_dataset, 'dialogue_id', n_splits=1, train_size=.75)


            # if using this review size of splits
            #train_dataset, test_dataset = train_dataset.train_test_split(train_size=0.85, shuffle=True, seed=SEED).values()  
            #dev_dataset, test_dataset = test_dataset.train_test_split(train_size=0.75, shuffle=True, seed=SEED).values()  

        elif size_data == 2:
            train_dataset = self.data_for_model['train']
            test_dataset = self.data_for_model['test']
            train_dataset, dev_dataset = self.splitter_by_groups(train_dataset, 'dialogue_id', n_splits=1, train_size=.75)
            #train_dataset, dev_dataset = test_dataset.train_test_split(train_size=0.75, shuffle=True, seed=SEED).values()  

        else:
            train_dataset = self.data_for_model['train']
            test_dataset = self.data_for_model['test']
            dev_dataset = self.data_for_model['dev']
        

        if subsetting:
            og_set = train_dataset[0]['labels'][:50]
            train_dataset = Subset(train_dataset, range(95))
            test_dataset = Subset(test_dataset, range(35))
            dev_dataset = Subset(dev_dataset, range(37))

            new_set = train_dataset[0]['labels'][:50]
            compare_tensors = torch.all(torch.eq(og_set, new_set))
            assert compare_tensors, "Subset does not correspond to original dataset"
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dev_dataset = dev_dataset


    #TODO: change workers to 12 when testing with gpu
    # we are not shuffling because we shuffled the dialogues before and this preserving the order of turns
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=8, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=8, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=8, num_workers=self.num_workers)

    @staticmethod
    def splitter_by_groups(dataset, col, n_splits=1, train_size=.8):

        gs = GroupShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=SEED)
        indexes = next(gs.split(dataset, groups=dataset[col]))
        bigger_dataset = dataset.select(indices=indexes[0])
        smaller_dataset = dataset.select(indices=indexes[1])
        return bigger_dataset, smaller_dataset
