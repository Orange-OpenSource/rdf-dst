from datasets import load_from_disk, dataset_dict
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from utils.predata_collate import PreDataCollator
import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes


class DialogueRDFData(LightningDataModule):
    """
    Preprocessor for DST prediction. Need previous system utterance, previous state, and
    user utterance to yield new state?
    """

    def __init__(self, tokenizer, data_dir: str='../san-fran-tito/', batch_size: int=6, data_type='SF'):
        super().__init__()
        self.max_len = 512
        self.data_dir = data_dir
        self.data_type = data_type
        self.tokenizer = tokenizer

        # initializing vars to track new data. This is ugly, I know
        data_keys = {"SF": {'history': 'dial', 'system': ['S', 'base'], 'user': ['U', 'hyp']},
                "DTSC": {'history': 'turns', 'system': ['output', 'transcript'], 'user': ['label', 'transcription']}}
        self.history = data_keys[self.data_type]['history']
        self.sys, self.s_trans= data_keys[self.data_type]['system']
        self.user, self.u_trans= data_keys[self.data_type]['user']

    def prepare_data(self):
        """
        SF has only a train split
        """
        rdf_dir = self.data_dir + 'rdf'
        # important to pass format so dataloader loads them as tensors and not lists
        txt2rdf = load_from_disk(rdf_dir).with_format("torch")
        collator = PreDataCollator(self.tokenizer, self.max_len, self.history)
        headers = txt2rdf['train'].features.keys()
        txt2rdf = self.flatten_data(txt2rdf, headers)

        if len(txt2rdf) == 1:
            train_dataset = txt2rdf['train']
            train_dataset = train_dataset.map(collator, remove_columns=self.history, num_proc=8, batched=True)  # num_proc == batches
            self.data_for_model = dataset_dict.DatasetDict({'train': train_dataset})

        elif len(txt2rdf) == 2:
            train_dataset = txt2rdf['train']
            test_dataset = txt2rdf['test']
            train_dataset = train_dataset.map(collator, remove_columns=self.history, num_proc=8, batched=True)  
            test_dataset = test_dataset.map(collator, remove_columns=self.history, num_proc=8, batched=True) 
            self.data_for_model = dataset_dict.DatasetDict({'train': train_dataset, 'test': test_dataset})

        elif len(txt2rdf) == 3:
            train_dataset = txt2rdf['train']
            test_dataset = txt2rdf['test']
            val_dataset = txt2rdf['dev']

            train_dataset = train_dataset.map(collator, remove_columns=self.history, num_proc=8, batched=True)  
            test_dataset = test_dataset.map(collator, remove_columns=self.history, num_proc=8, batched=True) 
            dev_dataset = dev_dataset.map(collator, remove_columns=self.history, num_proc=8, batched=True) 

            self.data_for_model = dataset_dict.DatasetDict({'train': train_dataset, 'test': test_dataset, 'dev': dev_dataset})

        else:
            raise Exception("No data splits provided")

    def setup(self):

        size_data = len(self.data_for_model)
        if size_data == 1:
            train_dataset = self.data_for_model['train']
            train_dataset, test_dataset = train_dataset.train_test_split(test_size=0.2, shuffle=True, seed=SEED).values()  
            dev_dataset, test_dataset = test_dataset.train_test_split(test_size=0.5, shuffle=True, seed=SEED).values()  

        elif size_data == 2:
            train_dataset = self.data_for_model['train']
            test_dataset = self.data_for_model['test']
            dev_dataset, test_dataset = test_dataset.train_test_split(test_size=0.5, shuffle=True, seed=SEED).values()  

        else:
            train_dataset = self.data_for_model['train']
            test_dataset = self.data_for_model['test']
            dev_dataset = self.data_for_model['dev']

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dev_dataset = dev_dataset

        #INFO collate_fn was passed to datasets and not loaders in prepare_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=16, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=16, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=16, num_workers=12)

    def flatten_data(self, dataset, headers):
        """

        txt2rdf: must be a dataset object
        headers: to remove added col
        """
        bad_cols = list(filter(lambda x: x != self.history, headers))
        dataset = dataset.map(self.data_cleaner, remove_columns=bad_cols)  # why isnt this removing the columns?
        dataset = dataset.remove_columns(bad_cols)
        return dataset

    def data_cleaner(self, dataset):
        clean_history = []
        for t in dataset[self.history]:
            new_turn = {'S': t[self.sys][self.s_trans],
                        'U': t[self.user][self.u_trans]}
            if self.data_type == 'SF':
                rdf_triplet = t[self.sys]['rdf-state']['triples']
                new_turn.setdefault('rdf-state', rdf_triplet)

            elif self.data_type == 'DTSC':  # rdf-state is stored in the usr in this case
                rdf_triplet = t[self.user]['rdf-state']['triples']
                new_turn.setdefault('rdf-state', rdf_triplet)

            clean_history.append(new_turn)
        dataset[self.history] = clean_history
        return dataset
