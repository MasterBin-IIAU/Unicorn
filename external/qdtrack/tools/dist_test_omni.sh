#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
EXP_NAME=$4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_omni.py $CONFIG $CHECKPOINT $EXP_NAME --launcher pytorch ${@:5}
