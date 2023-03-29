#!/bin/bash -e

#TODO: pip install and activate env

git clone --branch dev git@gitlab.tech.orange:NEPAL/task-oriented-dialogue/poc-rdf.git
POCRDFREPO=./poc-rdf
cp path_inserter.txt $POCRDFREPO
cd $POCRDFREPO
paths="$(cat path_inserter.txt)"
sed -i "/import os/a$paths" ./act2txt/sfxdial_rdf.py
sed -i "/import os/a$paths" ./act2txt/dstc2_rdf.py
sed -i "/import os/a$paths" ./act2txt/multiwoz_rdf.py

python ./act2txt/sfxdial_rdf.py -o ../sfx_rdf_data
python ./act2txt/multiwoz_rdf.py -o ../multiwoz_rdf_data
python ./act2txt/dstc2_rdf.py -o ../dstc2_rdf_data
#
git restore ./act2txt/sfxdial_rdf.py ./act2txt/dstc2_rdf.py ./act2txt/multiwoz_rdf.py
#
