# !/bin/bash -e

DEVICES="0"
#CUDA_VISIBLE_DEVICES=$DEVICES python main.py -epochs 5 --batch 8 -d multiwoz -workers 6 -store no
# FOR TESTING PURPOSES
CUDA_VISIBLE_DEVICES=$DEVICES python main.py -epochs 1 -d all -store yes -logger no
