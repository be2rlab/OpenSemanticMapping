# Open Semantic Mapping Evaluation Pipeline
## Contents:
- Concept-Graphs 
- Concept-Fusion (Todo)
- Open-Fusion (Todo)
- OpenMask3D
- OVSG (Todo)
- OVIR-3D (Todo)
- Semantic Abstraction (Todo)
- PLA (Todo)

## Concept-graphs:

### Build docker
```bash
git clone https://github.com/be2rlab/OpenSemanticMapping.git
cd OpenSemanticMapping
make build-concept-graphs
```

### Download Replica Datasets
```bash
bash ./datasets/scripts/download_replica.sh
```

### Download All Assets
```bash 
download from https://disk.yandex.ru/d/lAWeTD3SSNncaA
```

### Visualization
```bash
make prepare-terminal-for-visualization
```

### RUN
```bash
make run-concept-graphs
```

### Specific settings 
Inside the docker you can edit the export files in export folder, and then run one of them
```bash
bash /export/export_concept_graphs_light_ram.sh
```

## OpenMask3D:

### Build docker
```bash
make build-openmask3d
```

### Transform dataset
```
sudo python3 ./datasets/scripts/transformations/transform_to_openmask3d_format.py
```
Change DATASET_PATH variable in the transform_to_openmask3d_format.py scipt by the path to the dataset you want to transform.

### RUN
```bash
make run-openmask3d
```
