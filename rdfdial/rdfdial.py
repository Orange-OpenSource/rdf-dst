import os
import urllib
import re
import json

from pyarrow import json as pa_json
import datasets

_HOMEPAGE = "https://github.com/Orange-OpenSource/rdfdial/tree/e556a56fe82d19c3590fc6bd9f3ea1da39158025"

_ARCHIVES = {
    'dstc2': "https://github.com/Orange-OpenSource/rdfdial/raw/e556a56fe82d19c3590fc6bd9f3ea1da39158025/dstc2-rdf.tar.gz",
    'sfxdial': "https://github.com/Orange-OpenSource/rdfdial/raw/e556a56fe82d19c3590fc6bd9f3ea1da39158025/sfxdial-rdf.tar.gz",
    'multiwoz': "https://github.com/Orange-OpenSource/rdfdial/raw/e556a56fe82d19c3590fc6bd9f3ea1da39158025/multiwoz-rdf.tar.gz",
    'camrest-sim': "https://github.com/Orange-OpenSource/rdfdial/raw/e556a56fe82d19c3590fc6bd9f3ea1da39158025/camrest-sim-rdf.tar.gz",
    'multiwoz-sim': "https://github.com/Orange-OpenSource/rdfdial/raw/e556a56fe82d19c3590fc6bd9f3ea1da39158025/multiwoz-sim-rdf.tar.gz",
}

_BUNDLES = {
    "converted": ["dstc2","sfxdial","multiwoz"],
    "simulated": ["camrest-sim","multiwoz-sim"]
}


class rdfdial(datasets.ArrowBasedBuilder):
    VERSION = datasets.Version("2.2.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="bundle-converted",version=VERSION,
            description="Merge of all rdf converted datasets"),
        datasets.BuilderConfig(
            name="bundle-simulated",version=VERSION,
            description="Merge of all rdf simulated datasets"),
        datasets.BuilderConfig(
            name="dstc2",version=VERSION,
            description="DSTC2 converted to rdf format"),
        datasets.BuilderConfig(
            name="sfxdial",version=VERSION,
            description="sfxdial converted to rdf format"),
        datasets.BuilderConfig(
            name="multiwoz",version=VERSION,
            description="multiwoz2.3 converted to rdf format"),
        datasets.BuilderConfig(
            name="camrest-sim",version=VERSION,
            description="Synthetic dialogs on the Cambridge restaurant search domain"),
        datasets.BuilderConfig(
            name="multiwoz-sim",version=VERSION,
            description="Synthetic dialogs on the Multiwoz domains"),
    ]
    DEFAULT_CONFIG="all"

    def _info(self):
        features = datasets.Features(
                {
                    "dialogue_id": datasets.Value("string"),
                    "turns": [
                        {
                            "id": datasets.Value("int8"),
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "rdf-acts": [datasets.Value("string")],
                        }
                    ],
                    "states": [
                        {
                            "id": datasets.Value("int8"),
                            "multi_relations": datasets.Value("bool"),
                            "triples": [[datasets.Value("string")]],
                            "turn_ids": [datasets.Value("int8")],
                        }
                    ],
                })
        return datasets.DatasetInfo(
            description=self.config.description,
            # datasets.features.FeatureConnectors
            features=features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        if not self.config.name.startswith('bundle-'):
            archives = {self.config.name: _ARCHIVES[self.config.name]}
        else:
            archives = {x: _ARCHIVES[x] for x in _BUNDLES[self.config.name.split('-')[1]]}
        downloaded = dl_manager.download_and_extract(archives)
        paths = [os.path.join(v,'%s-rdf'%k) for k,v in downloaded.items()]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "paths": [os.path.join(p,'train.jsonl') for p in paths],
                    "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "paths": [os.path.join(p,'test.jsonl') for p in paths],
                    "split": "test"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "paths": [os.path.join(p,'validation.jsonl') for p in paths]     ,
                    "split": "validation"}
            )

        ]
    def _generate_tables(self, paths, split):
        idx = 0
        for p in paths:
            tbl = pa_json.read_json(p)
            yield idx,tbl
            idx += 1
