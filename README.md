
This repository contains the code to train, evaluate and analyse generative
models (T5 family) for Dialogue State Tracking using graph-based semantic
representation.

Results from our experiments are described in [results/description.md](results/description.md).

# Instructions


## Getting started

The scripts to download the datasets are in this repo in the `rdfdial/` directory.

Run setup to create a python virtual environment and install required dependencies.

```bash
./setup.sh
```

## Data Preprocessing, Training and Inference

```bash
./run.sh
```

Preprocessing of the converted RDF data, training, and inference is performed by the run script.

The code will look for the dataset loading script in `${HF_DATASETS_DIR}/rdfdial`.
When `HF_DATASETS_DIR` variable is not set, it'll default to the project's root directory.

## Evaluation

Based on this [evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation/blob/master/mwzeval/metrics.py)

Why is the most recent slot value that matters? See this:

[2015 paper](https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/44018.pdf)

## Tensorboard

Line to visualize results:
```bash
tensorboard --logdir tb_logs --bind_all
```

# License

Copyright (c) 2023 Orange

This code is released under the MIT license. See the LICENSE file for more information.
