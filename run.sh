#!/bin/bash -e

DEVICES="0"
CUDA_VISIBLE_DEVICES=$DEVICES python main.py -epochs 3

#rm -R ./sfx_rdf_data ./dstc2_rdf_data ./multiwoz_rdf_data
