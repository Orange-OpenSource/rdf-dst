from datasets import load_dataset, concatenate_datasets, load_from_disk, disable_caching
from torch.utils.data import DataLoader
import os
import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes
disable_caching()  # we have a lot of different collators that depend on data sets and experimental setup. We have to avoid reloading previous maps

class DialogueData:
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
            path = self.dataset + '_data'
            path = os.path.join('..', path)
            dialogue_data = load_from_disk(path).with_format("torch")  # path should be just multiwoz

            dialogue_data.cleanup_cache_files()
        else:
            dialogue_rdf = load_dataset("rdfdial", self.dataset, download_mode='force_redownload').with_format("torch")

            #dialogue_rdf.cleanup_cache_files()

            dialogue_rdf_data = concatenate_datasets([dialogue_rdf['validation'], dialogue_rdf['train'], dialogue_rdf['test']])  # splits are weird
            rdf_ids = set(dialogue_rdf_data['dialogue_id'])
            if self.dataset == "multiwoz":
                dialogue_data = load_dataset(self.dataset + '-convlab2', "v2.3", download_mode='force_redownload').with_format("torch")
            else:
                dialogue_data = load_dataset(self.dataset, download_mode='force_redownload').with_format("torch")

            #dialogue_data.cleanup_cache_files()

            if ('validation' not in dialogue_data.keys()) and ('test' not in dialogue_data.keys()):  
                all_data = dialogue_data['train']
                diag_id = {"multiwoz": 'dialogue_id', 'sfxdial': 'id', 'dstc2': 'session-id'}
                diag_id = diag_id[self.dataset]
                #all_data = all_data.filter(lambda x: x['dialogue_id'] in rdf_ids)
                all_data = all_data.filter(lambda x: x[diag_id] in rdf_ids)
                train_val = all_data.train_test_split(test_size=0.2)
                test_val = train_val['test'].train_test_split(test_size=0.5)
                dialogue_data.update({'train': train_val['train'], 'validation': test_val['train'], 'test': test_val['test']})

            else:
                all_data = concatenate_datasets([dialogue_data['validation'], dialogue_data['train'], dialogue_data['test']])  # splits are weird
                all_data = all_data.filter(lambda x: x['dialogue_id'] in rdf_ids)

                train_val = all_data.train_test_split(test_size=0.2)
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
                self.test_dataset = self.test_dataset.select(range(10))
            return {"test": self.test_dataloader()}

        self.train_dataset = self.dialogue_data['train'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['train'].column_names,
                                                             batched=True, load_from_cache_file=False)  
        self.validation_dataset = self.dialogue_data['validation'].map(self.collator, num_proc=8, remove_columns=self.dialogue_data['validation'].column_names,
                                                                       batched=True, load_from_cache_file=False) 

        if subsetting:
            self.train_dataset = self.train_dataset.select(range(45))
            self.validation_dataset = self.train_dataset.select(range(33))

        return {"train": self.train_dataloader(), "validation": self.validation_dataloader()}

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def validation_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
