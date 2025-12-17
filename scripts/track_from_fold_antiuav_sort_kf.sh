#!/bin/bash

DATASET_CFG="AntiUAV-val"
TRACKER="sort"
KALMAN_FORMAT="sort"
MIN_AREA=10
CONF_THRESH_LOW=0.05
CONF_THRESH=0.3
TRACK_BUFFER=60
INIT_TRACK_THRESH=0.4
SAVE_FOLDER="fromdet-KF"
MATCH_THRESH=0.7
DETECTED_FOLDER="./results_detected/antiuav-half-1600/val"
EVAL_YAML="AntiUAV"

# 执行跟踪命令
python track.py \
    --dataset "$DATASET_CFG" \
    --tracker "$TRACKER" \
    --kalman_format "$KALMAN_FORMAT" \
    --motion_model_path "$MOTION_MODEL_PATH" \
    --min_area "$MIN_AREA" \
    --conf_thresh_low "$CONF_THRESH_LOW" \
    --conf_thresh "$CONF_THRESH" \
    --match_thresh "$MATCH_THRESH" \
    --track_buffer "$TRACK_BUFFER" \
    --init_thresh "$INIT_TRACK_THRESH" \
    --save_folder "$SAVE_FOLDER" \
    --eval_yaml "$EVAL_YAML" \
    file \
    --detected_folder "$DETECTED_FOLDER"