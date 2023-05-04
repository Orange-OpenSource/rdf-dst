#!/bin/bash -e

DEVICES="0"
CUDA_VISIBLE_DEVICES=$DEVICES python main.py -epochs 3 -d multiwoz -store no
