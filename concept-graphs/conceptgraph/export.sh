export GSA_PATH=/home/sber20/dev/concept-graphs/third-party/Grounded-Segment-Anything
export LLAVA_PYTHON_PATH=/home/sber20/dev/concept-graphs/third-party/LLaVA
export LLAVA_MODEL_PATH=/home/sber20/dev/concept-graphs/third-party/LLaVA/weights/llava-v1.5-13b
export REPLICA_ROOT=/home/sber20/dev/SBER/data/Indoors/SBER/SBER_handheld/zed2/zed_node
export CG_FOLDER=/home/sber20/dev/concept-graphs
export REPLICA_CONFIG_PATH=${CG_FOLDER}/conceptgraph/dataset/dataconfigs/replica/replica.yaml
export SCENE_NAMES=office_sber_zed_depth
export SCENE_NAME=office_sber_zed_depth
export CLASS_SET=ram
export THRESHOLD=1.2


python scripts/run_slam_rgb.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 720 \
    --image_width 1280 \
    --stride 5 \
    --visualize
