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
TBD
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

