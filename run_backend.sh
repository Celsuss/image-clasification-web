#!/usr/bin/env bash

# start the backend

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    source venv/bin/activate
elif [ "$(expr substr $(uname -s) 1 9)" == "CYGWIN_NT" ]; then
    source venv/Scripts/activate
fi

cd ./src

export FLASK_ENV=development && flask run