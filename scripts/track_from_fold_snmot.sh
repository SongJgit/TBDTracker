#!/bin/bash

DATASET_CFG="snmot"
TRACKER="bytetrack"
# KALMAN_FORMAT="byteknet"
KALMAN_FORMAT="byte"
MIN_AREA=150
CONF_THRESH_LOW=0.1
CONF_THRESH=0.6
TRACK_BUFFER=30
# MOTION_MODEL_PATH="/data/Project/FIlters/filternet-dev/runs/SNMOT_msl50_0.1_CXCYWH/SIK_sNone_dNone_wd0.001_lr0.001_v0"
# MOTION_MODEL_PATH=""
INIT_TRACK_THRESH=0.6
SAVE_FOLDER="fromdet-KF"
# SAVE_DIR="./results_tracked/snmot-fromdet-OC-KNet"
DETECTED_FOLDER="/data/MOT_Datasets/SNMOT/test"

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
