#!/bin/bash

# Define parameters as uppercase variables
# CONFIG_PATH="cfg/eval_cfg/AntiUAV.yaml"
CONFIG_PATH="configs/eval_cfg/DanceTrack.yaml"
# CONFIG_PATH="cfg/eval_cfg/SNMOT.yaml"

# Run custom dataset evaluation script
python run_custom_dataset_eval.py \
    --config_path ${CONFIG_PATH}

# Check execution status
if [ $? -eq 0 ]; then
    echo "Custom dataset evaluation completed successfully"
else
    echo "Custom dataset evaluation failed"
    exit 1
fi
