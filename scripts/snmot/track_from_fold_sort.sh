
DATASET_CFG="soccernet"
TRACKER_CFG="sort"
DETECTED_FOLDER="/home/feng/songj_workspace/MOT_Datasets/SNMOT/test"
EVAL_YAML="SNMOT"
SAVE_FOLDER="fromdet-KF"

# 执行跟踪命令
python track.py \
    --dataset "$DATASET_CFG" \
    --tracker "$TRACKER_CFG" \
    --eval_yaml "$EVAL_YAML" \
    --save_folder "$SAVE_FOLDER" \
    file \
    --detected_folder "$DETECTED_FOLDER"
