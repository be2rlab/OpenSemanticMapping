
# Open Semantic Mapping Evaluation Pipeline
## Contents:
- Concept-Graphs 
- Concept-Fusion (Todo)
- Open-Fusion (Todo)
- OpenMask3D (Todo)
- OVSG (Todo)
- OVIR-3D (Todo)
- Semantic Abstraction (Todo)
- PLA (Todo)

### Build docker
```bash
cd concept-graphs
make build concept-graphs
```

## Dataset installation
You folder tree should be the same. Follow [habitat-sim](https://github.com/facebookresearch/habitat-sim?tab=readme-ov-file#installation) and [habitat-lab](https://github.com/facebookresearch/habitat-lab?tab=readme-ov-file#installation) installation guides.
```
habitat-lab/
    ...
habitat-sim/
    ...
be2r-habitat-data-generator/
    ...
data/
    hm3d_v0.2/
        minival/
            ...
    ReplicaCAD_dataset/
        ...
```

### HM3D
Follow [installation guide](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d). If you already have access to Matterport datasets:

```
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival_v0.2
```

After installation:
* improve `shader_type` field in `scene_dataset_config` to `material` or `phong` (lights reconfiguring don't work with other shader types)

### Replica CAD
Dataset might be installed in two ways. [Better one](https://aihabitat.org/datasets/replica_cad/):
```
# with conda install
python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path data/

# with source
python /path/to/habitat_sim/src_python/habitat_sim/utils/datasets_download.py --uids replica_cad_dataset --data-path data/
```

And another one:
```
git lfs install
huggingface-cli lfs-enable-largefiles .
git clone https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset -b main
```

Pay attention that we need ReplicaCAD **without** backed lights.

After installation:
* be careful, for some classes semantics annotation doesn't exist in basic dataset. But we can improve that. Take a look at `configs/ssd/replicaCAD_semantic_lexicon.json` for annotations and `urdf/{object}/{object}.ao_config.json` for possible improvements. In our case `cabinet.ao_config.json` starts with:
```
{
  "semantic_id": 18,
  "user_defined": {
    ...
  }
}
```

Our semantic annotaions for object in `urdf/` dir:
```
cabinet: 18
chestOfDrawers_01: 18
door{1/2/3}: 37
doubledoor: 37
fridge: 67
kitchen_counter: 2
kitchenCupboard_01: 94
```
There is no `ao_config` for doors in basic dataset. But we may create it. Unfortunately, we still haven't find ways to create semantics for whole doors in our scene (config describes only added doors). But it's possible to put door objects in same place as doors in scene. For example, add following text to `scene_instence.json` to get semantics for door:
```
"articulated_object_instances": [
    {
        "template_name": "door1",
        "fixed_base": true,
        "auto_clamp_joint_limits": true,
        "translation_origin": "COM",
        "motion_type": "DYNAMIC",
        "translation": [
        0.07893,
        2.0709,
        4.1143
        ],
        "rotation": [
        0, 
        1, 
        0, 
        0
        ]
    }
]
```