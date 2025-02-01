#!/usr/bin/env bash


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

accelerate launch --config_file /home/jackson/LLM-LinAtt/default_config.yaml src/train/train_attn.py --config "$CONFIG_FILE"
