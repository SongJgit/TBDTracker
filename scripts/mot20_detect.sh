#!/bin/bash

# Define parameters as uppercase variables
DATASET="./cfg/data_cfg/mot20-half-train.yaml"
DETECTOR="yolox"
YOLOX_EXP_FILE="./tracker/detectors/yolox_utils/yolox_x_mix_mot20_ch.py"
WEIGHTS="./weights/detector_weighs/yolox_x_MOT20_ablation.pt"
SAVE_IMAGES="--save_images"
SAVE_DIR="./results_detected/mot20-half/"

# Run detection script
python detect.py \
    --dataset ${DATASET} \
    --detector ${DETECTOR} \
    --yolox_exp_file ${YOLOX_EXP_FILE} \
    --weights ${WEIGHTS} \
    ${SAVE_IMAGES} \
    --save_dir ${SAVE_DIR}

# Check execution status
if [ $? -eq 0 ]; then
    echo "Detection completed successfully. Results saved to ${SAVE_DIR}"
