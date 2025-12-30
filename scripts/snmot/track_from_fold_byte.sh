#!/bin/bash

DATASET_CFG="soccernet"
TRACKER_CFG="bytetrack"
EVAL_YAML="SNMOT"
DETECTED_FOLDER="/home/feng/songj_workspace/MOT_Datasets/SNMOT/test"

SAVE_FOLDER="fromdet-KF"


# 执行跟踪命令
python test.py \
    --dataset "$DATASET_CFG" \
    --tracker "$TRACKER_CFG" \
    --eval_yaml "$EVAL_YAML" \
    --save_folder "$SAVE_FOLDER" \
    file \
    --detected_folder "$DETECTED_FOLDER"
