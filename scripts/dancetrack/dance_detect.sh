#!/bin/bash

# Define parameters as uppercase variables
DATASET="./configs/data_cfg/dancetrack-val.yaml"
DETECTOR="yolox"
WEIGHTS="./weights/detector_weights/yolox_x_dancetrack_ablation.pt"
YOLOX_EXP_FILE="./tracker/detectors/yolox_utils/yolox_x_ablation.py"
SAVE_IMAGES="--save_images"
SAVE_DIR="./results_detected/dance/"

# Run detection script
python detect.py \
    --dataset ${DATASET} \
    --detector ${DETECTOR} \
    --yolox_exp_file ${YOLOX_EXP_FILE} \
    --weights ${WEIGHTS} \
    ${SAVE_IMAGES} \
    --save_dir ${SAVE_DIR}
