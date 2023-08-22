#!/bin/bash -e

python3 -m venv viz-snake
source ./viz-snake/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e git+https://github.com/Zappandy/ecco-dst.git#egg=ecco
