#!/bin/bash -e

rm -Rf poc-rdf/
python main.py -epochs 3

#rm -R ./sfx_rdf_data ./dstc2_rdf_data ./multiwoz_rdf_data
