#!/bin/bash -e

python3 -m venv poc-rdf-env
source ./poc-rdf-env/bin/activate
pip install --upgrade pip
pip install -r poc-rdf_requirements.txt

git clone --branch dev git@gitlab.tech.orange:NEPAL/task-oriented-dialogue/poc-rdf.git
POCRDFREPO=./poc-rdf

cd $POCRDFREPO

PYTHONPATH=. python ./act2txt/sfxdial_rdf.py -o ../sfx_rdf_data
PYTHONPATH=. python ./act2txt/multiwoz_rdf.py -o ../multiwoz_rdf_data
PYTHONPATH=. python ./act2txt/dstc2_rdf.py -o ../dstc2_rdf_data

rm -Rf poc-rdf/
