#!/bin/bash

DATASET_CFG="dancetrack-val"
TRACKER="ocsort"
KALMAN_FORMAT="ocsort"
MIN_AREA=150
CONF_THRESH_LOW=0.1
CONF_THRESH=0.3
TRACK_BUFFER=30
INIT_TRACK_THRESH=0.7
MATCH_THRESH=0.3 # in ocsort, this is the iou_threshold
EVAL_YAML="DanceTrack"
SAVE_FOLDER="fromdet-KF"
DETECTED_FOLDER="./results_detected/dance/val"

# 执行跟踪命令
python track.py \
    --dataset "$DATASET_CFG" \
    --tracker "$TRACKER" \
    --kalman_format "$KALMAN_FORMAT" \
    --motion_model_path "$MOTION_MODEL_PATH" \
    --min_area "$MIN_AREA" \
    --conf_thresh_low "$CONF_THRESH_LOW" \
    --conf_thresh "$CONF_THRESH" \
    --track_buffer "$TRACK_BUFFER" \
    --init_thresh "$INIT_TRACK_THRESH" \
    --match_thresh "$MATCH_THRESH" \
    --eval_yaml "$EVAL_YAML" \
    --save_folder "$SAVE_FOLDER" \
    file \
    --detected_folder "$DETECTED_FOLDER"

