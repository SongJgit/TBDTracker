#!/bin/bash

# Define parameters as uppercase variables
DATASET="./cfg/data_cfg/AntiUAV-train.yaml"
DETECTOR="yolo12"
WEIGHTS="./weights/detector_weighs/yolo12_n_AntiUAV_1600_ablation.pt"
SAVE_IMAGES="--save_images"
IMG_SIZE=1600
SAVE_DIR="./results_detected/antiuav-half-1600/"

# Run detection script
python detect.py \
    --dataset ${DATASET} \
    --detector ${DETECTOR} \
    --img_size ${IMG_SIZE} \
    --weights ${WEIGHTS} \
    ${SAVE_IMAGES} \
    --save_dir ${SAVE_DIR}
