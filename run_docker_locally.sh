#!/usr/bin/env bash

source vars.sh
docker build -t $TAG .
docker run -p $PORT:$PORT -e PORT=$PORT $TAG