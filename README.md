# Instructions


## Getting started

The scripts to download the datasets are in this repo

USING KISSINGCAT NOW, must install requests then run orange\_certs.py while making sure AWS credentials for kissingcat are in your env as well as your profile.

INSTRUCTIONS WILL BE UPDATED ACCORDINGLY. Running run.sh is enough

```bash
git@gitlab.tech.orange:kissingcat/hf-datasets.git
```

The datasets variable ```HF_DATASETS_DIR``` points to this path and this is needed to run from VMs at Orange.

Run setup to install packages, download data, and convert it to RDF
```bash
./setup.sh
```

The installation should include this [repo](https://gitlab.tech.orange/morgan.veyret/deeper/-/tree/master/), which in the setup file is installed by the following command

```bash
pip install git+ssh://git@gitlab.tech.orange/morgan.veyret/deeper.git
```

This is not mandatory, but it allows us to run this project as a job using docker.

## Data Preprocessing, Training and Inference

```bash
./run.sh
```

Preprocessing of the converted RDF data, training, and inference is performed by the run script.


To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Evaluation

Based on this 

[evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation/blob/master/mwzeval/metrics.py)

Why is the most recent slot value that matters? See this:

[2015 paper](https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/44018.pdf)
## Tensorboard

Line to visualize results:
```bash
tensorboard --logdir tb_logs --bind_all
```
# License

Copyright (c) 2023 Orange

This code is released under the MIT license and Apache 2.0 license. See the LICENSE file for more information.
