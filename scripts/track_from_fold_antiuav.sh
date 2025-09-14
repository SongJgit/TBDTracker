#!/bin/bash

DATASET_CFG="AntiUAV-val"
TRACKER="bytetrack"
KALMAN_FORMAT="byteknet"
MIN_AREA=10
CONF_THRESH_LOW=0.05
CONF_THRESH=0.3
TRACK_BUFFER=60
MOTION_MODEL_PATH="/home/feng/songj_workspace/filternet-dev/runs/AntiUAV_msl50_0.05_CXCYWH/SIK_sNone_dNone_wd0_lr0.001_v0"
INIT_TRACK_THRESH=0.2
# SAVE_DIR="./results_tracked/antiuav-fromdet-byteKF1600"
SAVE_FOLDER="fromdet-SIKNet-005"
DETECTED_FOLDER="./results_detected/antiuav-half-1600/val"

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
    --save_images \
    --save_folder "$SAVE_FOLDER" \
    file \
    --detected_folder "$DETECTED_FOLDER"
