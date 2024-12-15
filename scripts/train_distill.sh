#!/usr/bin/env bash

# Usage: ./run_training.sh config.yaml

CONFIG_FILE="$1"

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No configuration file supplied."
    echo "Usage: ./train_distill.sh config.yaml"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' does not exist."
    exit 1
fi

python3 -m src.train.train_attn --config "$CONFIG_FILE"
