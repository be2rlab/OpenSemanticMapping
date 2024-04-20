#!/bin/bash

export GSA_PATH=/tmp/Grounded-Segment-Anything
export LLAVA_PYTHON_PATH=/tmp/LLaVA
export LLAVA_MODEL_PATH=/assets/llm/llava-v1.5-13b
export LLAVA_CKPT_PATH=/assets/llm/llava-v1.5-13b
export REPLICA_ROOT=/data/generated/replica_cad/
export CG_FOLDER=/opt/src
export REPLICA_CONFIG_PATH=${CG_FOLDER}/conceptgraph/dataset/dataconfigs/replica/replica.yaml
export OPENAI_API_KEY=""
export SCENE_NAMES="v3_sc0_staging_00/default_lights_0/"
export SCENE_NAME="v3_sc0_staging_00/default_lights_0/"
export CLASS_SET=none
export THRESHOLD=1.2

python ${CG_FOLDER}/conceptgraph/scripts/generate_gsa_results.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set $CLASS_SET \
    --stride 5

python ${CG_FOLDER}/conceptgraph/slam/cfslam_pipeline_batch.py \
    dataset_root=$REPLICA_ROOT \
    dataset_config=$REPLICA_CONFIG_PATH \
    stride=5 \
    scene_id=$SCENE_NAME \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.95 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.1 \
    gsa_variant=none \
    class_agnostic=True \
    skip_bg=True \
    max_bbox_area_ratio=0.5 \
    save_suffix=overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub \
    merge_interval=20 \
    merge_visual_sim_thresh=0.8 \
    merge_text_sim_thresh=0.8 \
    save_objects_all_frames=True


# python ${CG_FOLDER}/conceptgraph/scripts/animate_mapping_interactive.py --input_folder $REPLICA_ROOT/$SCENE_NAME/objects_all_frames/<folder_name>
python ${CG_FOLDER}/conceptgraph/scripts/visualize_cfslam_results.py --result_path ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz
