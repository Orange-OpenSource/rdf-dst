#!/bin/bash -e
POCRDFREPO=$1
POCRDFREPO=$HOME/Documents/poc-rdf
cp path_inserter.txt $POCRDFREPO
cd $POCRDFREPO
paths="$(cat path_inserter.txt)"
sed -i "/import os/a$paths" ./act2txt/sfxdial_rdf.py
sed -i "/import os/a$paths" ./act2txt/dstc2_rdf.py
sed -i "/import os/a$paths" ./act2txt/multiwoz_rdf.py

python ./act2txt/sfxdial_rdf.py -o ../sfx_rdf_data
python ./act2txt/multiwoz_rdf.py -o ../multiwoz_rdf_data
python ./act2txt/dstc2_rdf.py -o ../dstc2_rdf_data

# returning to home andres' dir
cd -
python main.py -epochs 1
