#!/usr/bin/env bash

# create virtual env and install packages

python3 -m venv ./venv

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    source venv/bin/activate
elif [ "$(expr substr $(uname -s) 1 9)" == "CYGWIN_NT" ]; then
    source venv/Scripts/activate
fi

python -m pip install --upgrade pip

pip install -r requirements.txt