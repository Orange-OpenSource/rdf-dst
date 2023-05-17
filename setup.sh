#!/bin/bash -e

python3 -m venv dst-snake
source ./dst-snake/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python old_orange_certs.py
