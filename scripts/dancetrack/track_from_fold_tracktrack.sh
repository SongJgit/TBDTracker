#!/bin/bash

DATASET_CFG="dancetrack-val"
TRACKER_CFG="tracktrack"
EVAL_YAML="DanceTrack"
DETECTED_FOLDER="./results_detected/dance/val"

SAVE_FOLDER="fromdet-KF"


# 执行跟踪命令
python track.py \
    --dataset "$DATASET_CFG" \
    --tracker "$TRACKER_CFG" \
    --eval_yaml "$EVAL_YAML" \
    --save_folder "$SAVE_FOLDER" \
    file \
    --detected_folder "$DETECTED_FOLDER"
