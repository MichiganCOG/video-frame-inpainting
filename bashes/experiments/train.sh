#!/bin/bash

DEFAULT_ARGS_PATH="$1"
EXTRA_ARGS_PATH="$2"

# Extract args (ignoring lines that start with #)
DEFAULT_ARGS=`cat $DEFAULT_ARGS_PATH | grep -v '^#'`
EXTRA_ARGS=`cat $EXTRA_ARGS_PATH | grep -v '^#'`

python train.py $DEFAULT_ARGS $EXTRA_ARGS
