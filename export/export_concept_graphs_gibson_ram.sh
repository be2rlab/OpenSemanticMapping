#!/bin/bash

export GSA_PATH=/tmp/Grounded-Segment-Anything
export LLAVA_PYTHON_PATH=/tmp/LLaVA
export LLAVA_MODEL_PATH=/assets/llm/llava-v1.5-13b
export LLAVA_CKPT_PATH=/assets/llm/llava-v1.5-13b
export REPLICA_ROOT=/data/Datasets/Gibson
export CG_FOLDER=/opt/src
export REPLICA_CONFIG_PATH=${CG_FOLDER}/conceptgraph/dataset/dataconfigs/light/light.yaml
export OPENAI_API_KEY=""
export SCENE_NAMES=Adrian
export SCENE_NAME=Adrian
export CLASS_SET=ram
export THRESHOLD=1.2

# CLASS_SET=ram
# python ${CG_FOLDER}/conceptgraph/scripts/generate_gsa_results.py \
#     --dataset_root $REPLICA_ROOT \
#     --dataset_config $REPLICA_CONFIG_PATH \
#     --scene_id $SCENE_NAME \
#     --class_set $CLASS_SET \
#     --box_threshold 0.2 \
#     --text_threshold 0.2 \
#     --stride 5 \
#     --add_bg_classes \
#     --accumu_classes \
#     --exp_suffix withbg_allclasses

# python ${CG_FOLDER}/conceptgraph/slam/cfslam_pipeline_batch.py \
#     dataset_root=$REPLICA_ROOT \
#     dataset_config=$REPLICA_CONFIG_PATH \
#     stride=5 \
#     scene_id=$SCENE_NAME \
#     spatial_sim_type=overlap \
#     mask_conf_threshold=0.25 \
#     match_method=sim_sum \
#     sim_threshold=${THRESHOLD} \
#     dbscan_eps=0.1 \
#     gsa_variant=ram_withbg_allclasses \
#     skip_bg=False \
#     max_bbox_area_ratio=0.5 \
#     save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1 \
#     save_objects_all_frames=True


python ${CG_FOLDER}/conceptgraph/scripts/visualize_cfslam_results.py --result_path ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_post.pkl.gz
